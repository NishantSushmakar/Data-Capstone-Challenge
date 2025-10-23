# Validation Set Segmentation Mask Extraction

## Overview

Extracts segmentation masks from a trained coral segmentation model for validation set images. Used in the interpretability pipeline (EfficientNet + Random Forest + LIME).

## Output Files

The script generates two folders:

1. **`binary_masks/`** - Binary coral masks
   - 255 = coral region
   - 0 = background
   - Purpose: Feature extraction (next step)

2. **`segmentation_masks/`** - 3-class segmentation predictions
   - 0 = background
   - 1 = healthy coral
   - 2 = bleached coral
   - Purpose: Compare with Random Forest predictions later
   - Note: Pixel values are 0-2, so images appear black when viewed directly

## Usage

```python
python extract_val_segmentation.py
```

Default path configuration:
- Model path: `best_iou_model.pth`
- Dataset split: `dataset_split.json`
- Image directory: `images/`
- Output directory: `val_segmentation_outputs/`

To modify paths, adjust the parameters in the `extract_validation_segmentation()` function.

## Dependencies

- PyTorch
- PIL (Pillow)
- NumPy
- tqdm

## Workflow

1. Load validation set image list
2. Load trained EfficientNet segmentation model
3. Run inference on each validation image
4. Save binary masks and 3-class masks
5. Output files ready for interpretability analysis