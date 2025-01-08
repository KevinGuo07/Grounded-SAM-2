import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import argparse

'''
This program is used to predict the inference based on mask in first frame in video/segmented images
'''

# example:
# python gsam2_mask_tracking.py --text_prompt "red block. gripper." --input_dir "input" --output_dir "outputs/grip_mask"
# Init input hyperparameters, notice the input images should be .jpg
def parse_args():
    parser = argparse.ArgumentParser(description="Process a folder of images with GroundingDINO and SAM2")
    parser.add_argument("--text_prompt", type=str, default="gripper. block.", help="Text prompt for the model.")
    parser.add_argument("--input_dir", type=str, default="notebooks/videos/grip", help="Path to the input images folder.")
    return parser.parse_args()


args = parse_args()
TEXT_PROMPT = args.text_prompt
INPUT_DIR = args.input_dir


# latent hyperparameters
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
'''
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
'''
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "IDEA-Research/grounding-dino-tiny"  # from huggingface
processor = AutoProcessor.from_pretrained(model_id)


# based on instruction given by https://huggingface.co/IDEA-Research/grounding-dino-tiny
# build sam2, grounding_dino, and video predictor
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)
video_predictor = build_sam2_video_predictor(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT)
inference_state = video_predictor.init_state(video_path=INPUT_DIR)


# prepare the input images
frame_names = [
    p for p in os.listdir(INPUT_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

img_path = os.path.join(INPUT_DIR, frame_names[ann_frame_idx])
image = Image.open(img_path)


# run grounded dino
inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    target_sizes=[image.size[::-1]]
)

# process the detection results
input_boxes = results[0]["boxes"].cpu().numpy()
OBJECTS = results[0]["labels"]

# prompt SAM 2 image predictor to get the mask for the object
sam2_predictor.set_image(np.array(image.convert("RGB")))
masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

# convert masks shape
if masks.ndim == 3:
    masks = masks[None]
    scores = scores[None]
    logits = logits[None]
elif masks.ndim == 4:
    masks = masks.squeeze(1)


for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
    labels = np.ones((1), dtype=np.int32)
    _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=object_id,
        mask=mask
    )
    # region function to be achieved
    # mask in inference, while this is not achieved yet, can be ignored
    consolidated_out = video_predictor._consolidate_temp_output_across_obj(
        inference_state,
        frame_idx=ann_frame_idx,
        is_cond=True,
        run_mem_encoder=False,
        consolidate_at_video_res=True,
    )

    _, video_res_masks = video_predictor._get_orig_video_res_output(
        inference_state, consolidated_out["pred_masks_video_res"]
    )

    # get the mask
    obj_mask = video_res_masks[object_id - 1]
    obj_mask_np = obj_mask.cpu().numpy()
    # endregion

video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

save_dir = "./tracking_masks_results"
save_json_dir = "./tracking_json_results"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(save_json_dir):
    os.makedirs(save_json_dir)

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(INPUT_DIR, frame_names[frame_idx]))

    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)


    merged_mask = np.zeros_like(masks[0], dtype=np.int32)
    for i, mask in enumerate(masks):
        merged_mask[mask > 0] = object_ids[i]

    mask_path = os.path.join(save_dir, f"frame_{frame_idx:05d}_classified.npy")
    np.save(mask_path, merged_mask)

    json_save_path = os.path.join(save_json_dir, f"frame_{frame_idx:05d}_masks.json")


    mask_rles = []
    for obj_id in np.unique(merged_mask):
        if obj_id == 0:  # skip the background
            continue
        # get current mask
        single_mask = (merged_mask == obj_id).astype(np.uint8)
        rle = single_mask_to_rle(single_mask)
        mask_rles.append(rle)

    json_data = {
        "labels": list(OBJECTS),
        "segmentation": mask_rles
    }

    # save as json
    with open(json_save_path, "w") as json_file:

        json.dump(json_data, json_file, indent=4)


