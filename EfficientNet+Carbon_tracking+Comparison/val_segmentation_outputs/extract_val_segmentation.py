"""
Extract Segmentation Outputs for Validation Set Only
For Interpretability Pipeline (EfficientNet + Random Forest + LIME)

Outputs:
- binary_masks/: Binary coral masks (255=coral, 0=background)
  → Use this for feature extraction (next step)

- segmentation_masks/: EfficientNet's 3-class predictions (0=bg, 1=healthy, 2=bleached)
  → Use this for comparison with Random Forest predictions later
  → Note: Pixel values are 0-2, so images look black when viewed directly
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm
from pathlib import Path

# Import model from improved_model.py
from improved_model import ImprovedCoralModel


def extract_validation_segmentation(
    model_path=r'E:\MyFiles\Tue_Courses\Capstone_Data_Challenge\Data-Capstone-Challenge\improved_model\best_iou_model.pth',
    dataset_split_json=r'E:\MyFiles\Tue_Courses\Capstone_Data_Challenge\Data-Capstone-Challenge\improved_model\dataset_split.json',
    image_dir=r'E:\MyFiles\Tue_Courses\Capstone_Data_Challenge\data\images',
    output_dir=r'E:\MyFiles\Tue_Courses\Capstone_Data_Challenge\val_segmentation_outputs',
    target_size=512,
    device='cuda'
):
    """
    Extract segmentation masks for validation set images

    Outputs:
        - binary_masks/: Binary coral masks (255=coral, 0=background)
        - segmentation_masks/: 3-class masks (0=bg, 1=healthy, 2=bleached)
    """

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset split
    with open(dataset_split_json, 'r') as f:
        dataset_split = json.load(f)
    val_ids = dataset_split['val_ids']
    print(f"Validation set: {len(val_ids)} images")

    # Load model
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    backbone = checkpoint.get('config', {}).get('backbone', 'tf_efficientnet_b2_ns')
    model = ImprovedCoralModel(backbone_name=backbone, num_classes=3, pretrained=False)

    # Filter out unexpected keys (total_ops, total_params)
    state_dict = checkpoint['model_state_dict']
    state_dict = {k: v for k, v in state_dict.items()
                  if not ('total_ops' in k or 'total_params' in k)}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    if 'val_iou' in checkpoint:
        print(f"Model IoU: {checkpoint['val_iou']:.4f}")

    # Create output directories
    output_dir = Path(output_dir)
    binary_dir = output_dir / 'binary_masks'
    seg_dir = output_dir / 'segmentation_masks'

    binary_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    print(f"\nProcessing {len(val_ids)} validation images...")

    with torch.no_grad():
        for img_id in tqdm(val_ids, desc='Extracting segmentation'):
            # Load image
            img_path = os.path.join(image_dir, f'{img_id}.jpg')
            if not os.path.exists(img_path):
                print(f"Warning: {img_id}.jpg not found")
                continue

            # Preprocess
            image = Image.open(img_path).convert('RGB')
            image = image.resize((target_size, target_size), Image.BILINEAR)
            image_array = np.array(image, dtype=np.float32) / 255.0

            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).to(device)
            image_tensor = (image_tensor - mean) / std
            image_tensor = image_tensor.unsqueeze(0)

            # Get predictions
            outputs = model(image_tensor)
            seg_logits = outputs['segmentation']
            seg_pred = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy()

            # Binary mask (coral vs background)
            binary_mask = (seg_pred > 0).astype(np.uint8) * 255

            # Save masks
            Image.fromarray(binary_mask).save(binary_dir / f'{img_id}_coral_mask.png')
            Image.fromarray(seg_pred.astype(np.uint8)).save(seg_dir / f'{img_id}_segmentation.png')

    print(f"\nDone! Outputs saved to:")
    print(f"  Binary masks: {binary_dir}")
    print(f"  Segmentation: {seg_dir}")
    print(f"\nNext step: Use binary_masks/ for feature extraction")


if __name__ == "__main__":
    extract_validation_segmentation()
