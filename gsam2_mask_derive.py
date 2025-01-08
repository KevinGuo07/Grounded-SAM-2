import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import argparse


# example:
# python gsam2_mask_derive.py --text_prompt "red block. gripper." --input_dir "input" --output_dir "outputs/grip_mask"

# Init input hyperparameters, notice the input images should be .jpg
def parse_args():
    parser = argparse.ArgumentParser(description="Process a folder of images with GroundingDINO and SAM2")
    parser.add_argument("--text_prompt", type=str,
                        default="gripper. block.", help="Text prompt for the model.")
    parser.add_argument("--input_dir", type=str, default="image_jpg", help="Path to the input images folder.")
    parser.add_argument("--output_dir", type=str, default="outputs/grip_mask", help="Directory to save output results.")
    return parser.parse_args()


args = parse_args()
TEXT_PROMPT = args.text_prompt
INPUT_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)

# create output path for 2 formats
MASK_OUTPUT_DIR = OUTPUT_DIR / "masks_numpy"
JSON_OUTPUT_DIR = OUTPUT_DIR / "masks_json"
MERGED_MASK_OUTPUT_DIR = OUTPUT_DIR / "merged_masks_numpy"
MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MERGED_MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# latent hyperparameters
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# build sam2 and grounding_dino
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# Get rle for certain mask
def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

#region save function for different format
def save_origin_masks(masks, img_file, output_dir):
    mask_save_path = output_dir / f"{img_file.stem}_mask.npy"
    np.save(mask_save_path, masks)
    print(f"Saved NPY to {mask_save_path}")
    return mask_save_path

def save_merged_masks(masks, img_file, output_dir):
    merged_mask_save_path = output_dir / f"{img_file.stem}_merged_mask.npy"
    mask_array = masks.squeeze(1).astype("uint8")
    print("Mask shape:", mask_array.shape)
    merged_mask = np.zeros(mask_array.shape[1:], dtype=np.int32)
    for i, single_mask in enumerate(mask_array, start=1):
        merged_mask[single_mask == 1] = i
    np.save(merged_mask_save_path, merged_mask)
    print(f"Saved merged NPY to {merged_mask_save_path}")
    return merged_mask_save_path

def save_json_file(img_file, labels, input_boxes, scores, masks, output_dir, single_mask_to_rle):
    json_save_path = output_dir / f"{img_file.stem}_result.json"
    mask_array = masks.squeeze(1).astype("uint8")
    mask_save_path = output_dir / f"{img_file.stem}_mask.npy"
    mask_rles = [single_mask_to_rle(mask) for mask in mask_array]
    json_data = {
        "file_name": str(img_file),
        "labels": labels,
        "boxes": input_boxes.tolist(),
        "segmentation": mask_rles,
        "scores": scores.tolist(),
        "mask_path": str(mask_save_path),
    }
    with open(json_save_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"Saved JSON to {json_save_path}")
    return json_save_path

#endregion

# process all the images in input path
for img_file in INPUT_DIR.glob("*.jpg"):
    print(f"Processing image: {img_file}")
    image_source, image = load_image(str(img_file))
    sam2_predictor.set_image(image_source)

    # predict results in grounding_dino from given text and images
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # process the box prompt for SAM 2, where text prompt can't be directly used
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # FIXME: figure how does this influence the G-DINO model
    torch.autocast(device_type="cuda", dtype=torch.float32).__enter__()

    # predict masks in sam2 from given boxes
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    '''
    data save for 3 formats:    seperate mask(n*1*H*W)
                                merged mask(H*W)
                                json(containing path, name, box, label, rle)
    '''

    # origin masks
    save_origin_masks(masks, img_file, MASK_OUTPUT_DIR)

    # merged masks in one mask
    save_merged_masks(masks, img_file, MERGED_MASK_OUTPUT_DIR)

    # save as json file
    save_json_file(img_file, labels, input_boxes, scores, masks, JSON_OUTPUT_DIR, single_mask_to_rle)


