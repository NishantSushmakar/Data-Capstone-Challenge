# Import required libraries
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import argparse


# Set random seed for reproducibility
np.random.seed(42)


def extract_patch_with_boundary_padding(image, mask, x, y, patch_size=1024, padding_mode='constant'):
    """
    Extract a patch at given coordinates, padding only if it goes beyond image boundaries.
    This is much more elegant than padding the entire image first!
    
    Args:
        image: Original image (H, W, 3)
        mask: Corresponding mask (H, W, 3)
        x, y: Top-left coordinates for patch extraction
        patch_size: Target patch size
        padding_mode: Padding mode ('constant' or 'reflect')
    
    Returns:
        image_patch: Padded image patch (patch_size, patch_size, 3)
        mask_patch: Padded mask patch (patch_size, patch_size, 3)
        padding_info: Information about applied padding
    """
    img_height, img_width = image.shape[:2]
    
    # Calculate the actual crop region (may extend beyond image)
    x_end = x + patch_size
    y_end = y + patch_size
    
    # Calculate how much we can actually crop from the image
    actual_x_start = max(0, x)
    actual_y_start = max(0, y)
    actual_x_end = min(img_width, x_end)
    actual_y_end = min(img_height, y_end)
    
    # Extract the actual available region
    actual_image_crop = image[actual_y_start:actual_y_end, actual_x_start:actual_x_end]
    actual_mask_crop = mask[actual_y_start:actual_y_end, actual_x_start:actual_x_end]
    
    actual_crop_height, actual_crop_width = actual_image_crop.shape[:2]
    
    # Calculate padding needed
    pad_top = actual_y_start - y  # Padding needed at top (if y < 0)
    pad_left = actual_x_start - x  # Padding needed at left (if x < 0)
    pad_bottom = patch_size - actual_crop_height - pad_top  # Padding needed at bottom
    pad_right = patch_size - actual_crop_width - pad_left  # Padding needed at right
    
    padding_info = {
        'top': pad_top,
        'bottom': pad_bottom,
        'left': pad_left,
        'right': pad_right,
        'original_crop_size': (actual_crop_width, actual_crop_height),
        'requested_coords': (x, y),
        'actual_coords': (actual_x_start, actual_y_start)
    }
    
    # Apply padding to reach target patch size
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        if padding_mode == 'constant':
            padded_image = np.pad(actual_image_crop,
                                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                mode='constant', constant_values=0)
            padded_mask = np.pad(actual_mask_crop,
                               ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                               mode='constant', constant_values=0)
        elif padding_mode == 'reflect':
            padded_image = np.pad(actual_image_crop,
                                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                mode='reflect')
            padded_mask = np.pad(actual_mask_crop,
                               ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                               mode='reflect')
        else:
            raise ValueError(f"Unsupported padding mode: {padding_mode}")
    else:
        padded_image = actual_image_crop
        padded_mask = actual_mask_crop
    
    # Verify final size
    assert padded_image.shape == (patch_size, patch_size, 3), f"Image patch shape: {padded_image.shape}"
    assert padded_mask.shape == (patch_size, patch_size, 3), f"Mask patch shape: {padded_mask.shape}"
    
    return padded_image, padded_mask, padding_info

def process_segmentation_crop_then_pad_silent(images_dir, masks_dir, output_dir='coral_dataset_crop_pad', 
                                            patch_size=1024, overlap_ratio=0.5, 
                                            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                            padding_mode='constant', max_images=None, verbose=False):
    """
    Process segmentation dataset using crop-then-pad approach with minimal output.
    Fixes VS Code notebook output overflow issues.
    
    Args:
        images_dir: Path to images directory
        masks_dir: Path to masks directory  
        output_dir: Output directory for processed dataset
        patch_size: Size of patches (default 1024)
        overlap_ratio: Overlap ratio (default 0.5 for 50% overlap)
        train_ratio: Ratio for training set (default 0.7)
        val_ratio: Ratio for validation set (default 0.15)
        test_ratio: Ratio for test set (default 0.15)
        padding_mode: Padding mode for boundary patches ('constant' recommended)
        max_images: Maximum number of images to process (None for all)
        verbose: If True, shows detailed progress (may cause VS Code issues)
    
    Returns:
        Dictionary with processing statistics and results
    """
    import os
    import json
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split
    import sys
    
    if verbose:
        print(f"CROP-THEN-PAD PROCESSING")
        print("="*70)
        print(f"Images directory: {images_dir}")
        print(f"Masks directory: {masks_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Patch size: {patch_size}×{patch_size}")
        print(f"Overlap: {overlap_ratio*100:.0f}%")
        print(f"Strategy: Crop-then-pad (silent mode)")
    
    # Get all image files and create splits
    image_files = sorted(list(Path(images_dir).glob("*.jpg")))
    if max_images:
        image_files = image_files[:max_images]
    
    if verbose:
        print(f"Found {len(image_files)} images to process")
    
    # Create image-level splits
    if len(image_files) >= 3:
        train_images, temp_images = train_test_split(
            image_files, test_size=(val_ratio + test_ratio), random_state=42, shuffle=True
        )
        
        if len(temp_images) >= 2:
            val_test_ratio = val_ratio / (val_ratio + test_ratio)
            val_images, test_images = train_test_split(
                temp_images, test_size=(1 - val_test_ratio), random_state=42, shuffle=True
            )
        else:
            val_images = temp_images
            test_images = []
    else:
        train_images = image_files[:-2] if len(image_files) > 2 else image_files
        val_images = [image_files[-2]] if len(image_files) > 1 else []
        test_images = [image_files[-1]] if len(image_files) > 2 else []
    
    splits = {'train': train_images, 'val': val_images, 'test': test_images}
    
    # Create output directories
    for split_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split_name, 'masks'), exist_ok=True)
    
    # Statistics tracking
    stats = {
        'total_images': len(image_files),
        'splits': {split: {'images': len(splits[split]), 'patches': 0, 'boundary_patches': 0, 'full_patches': 0} 
                  for split in ['train', 'val', 'test']},
        'total_patches': 0,
        'total_boundary_patches': 0,
        'total_full_patches': 0,
        'processing_info': {
            'patch_size': patch_size,
            'overlap_ratio': overlap_ratio,
            'padding_mode': padding_mode,
            'strategy': 'crop_then_pad_silent',
            'boundary_padding_only': True
        },
        'errors': []
    }
    
    all_metadata = {'train': [], 'val': [], 'test': []}
    
    # Process each split with minimal output
    for split_name, split_images in splits.items():
        if len(split_images) == 0:
            continue
            
        if verbose:
            print(f"\\nProcessing {split_name} split ({len(split_images)} images)...")
        
        split_patch_count = 0
        split_boundary_patches = 0
        split_full_patches = 0
        
        # Use tqdm with minimal output or disable for large datasets
        progress_bar = tqdm(split_images, desc=f"{split_name}", 
                           disable=not verbose, 
                           file=sys.stdout if verbose else open(os.devnull, 'w'))
        
        for image_path in progress_bar:
            try:
                mask_path = Path(masks_dir) / f"{image_path.stem}.png"
                
                if not mask_path.exists():
                    stats['errors'].append(f"Mask not found for {image_path.name}")
                    continue
                
                # Load original images (no pre-padding!)
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                mask = cv2.imread(str(mask_path))
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                
                # Extract patches using crop-then-pad approach (silent)
                patches = create_patches_crop_then_pad_silent(
                    image, mask,
                    patch_size=patch_size,
                    overlap_ratio=overlap_ratio,
                    padding_mode=padding_mode
                )
                
                # Save ALL patches
                for patch_idx, patch in enumerate(patches):
                    img_patch = patch['image_patch']
                    mask_patch = patch['mask_patch']
                    filename = f"{image_path.stem}_patch_{patch_idx:04d}"
                    
                    # Ensure uint8 format
                    if img_patch.dtype != np.uint8:
                        if img_patch.max() <= 1.0:
                            img_patch = (img_patch * 255).astype(np.uint8)
                        else:
                            img_patch = img_patch.astype(np.uint8)
                    
                    if mask_patch.dtype != np.uint8:
                        mask_patch = mask_patch.astype(np.uint8)
                    
                    # Save patches
                    img_save_path = os.path.join(output_dir, split_name, 'images', f"{filename}.jpg")
                    cv2.imwrite(img_save_path, cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
                    
                    mask_save_path = os.path.join(output_dir, split_name, 'masks', f"{filename}.png")
                    cv2.imwrite(mask_save_path, cv2.cvtColor(mask_patch, cv2.COLOR_RGB2BGR))
                    
                    # Count boundary vs full patches
                    if patch['is_boundary_patch']:
                        split_boundary_patches += 1
                    else:
                        split_full_patches += 1
                    
                    # Store metadata
                    all_metadata[split_name].append({
                        'filename': filename,
                        'original_image': image_path.name,
                        'original_mask': mask_path.name,
                        'patch_coordinates': patch['coordinates'],
                        'is_boundary_patch': patch['is_boundary_patch'],
                        'padding_info': patch['padding_info'],
                        'strategy': 'crop_then_pad_silent',
                        'split': split_name
                    })
                    
                    split_patch_count += 1
                
                stats['splits'][split_name]['patches'] = split_patch_count
                stats['splits'][split_name]['boundary_patches'] = split_boundary_patches
                stats['splits'][split_name]['full_patches'] = split_full_patches
                stats['total_patches'] += len(patches)
                
                # Update progress bar description periodically
                if not verbose and split_patch_count % 100 == 0:
                    progress_bar.set_description(f"{split_name}: {split_patch_count} patches")
                
            except Exception as e:
                error_msg = f"Error processing {image_path.name}: {str(e)}"
                stats['errors'].append(error_msg)
                if verbose:
                    print(f"Error: {error_msg}")
        
        progress_bar.close()
        
        if verbose:
            print(f"  {split_name} completed: {split_patch_count:,} patches")
            print(f"    Full patches: {split_full_patches:,}")
            print(f"    Boundary patches: {split_boundary_patches:,}")
    
    # Calculate totals
    stats['total_boundary_patches'] = sum(stats['splits'][s]['boundary_patches'] for s in ['train', 'val', 'test'])
    stats['total_full_patches'] = sum(stats['splits'][s]['full_patches'] for s in ['train', 'val', 'test'])
    
    # Save metadata
    metadata_file = os.path.join(output_dir, 'dataset_info.json')
    full_metadata = {
        'dataset_info': {
            'creation_date': str(pd.Timestamp.now()),
            'strategy': 'crop_then_pad_silent',
            'advantages': [
                'More memory efficient',
                'Only pads boundary patches',
                'Preserves all original information',
                'Faster processing',
                'Silent mode prevents VS Code output issues'
            ]
        },
        'statistics': stats,
        'metadata_by_split': all_metadata
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(full_metadata, f, indent=2)
    
    # Final summary (always shown)
    print(f"\\n PROCESSING COMPLETE!")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total patches: {stats['total_patches']:,}")
    print(f"  Full patches: {stats['total_full_patches']:,} (no padding)")
    print(f"  Boundary patches: {stats['total_boundary_patches']:,} (padded)")
    for split in ['train', 'val', 'test']:
        print(f"  {split.capitalize()}: {stats['splits'][split]['patches']:,} patches")
    print(f"  Output: {output_dir}")
    print(f"  Errors: {len(stats['errors'])}")
    
    return full_metadata


def create_patches_crop_then_pad_silent(image, mask, patch_size=1024, overlap_ratio=0.5, padding_mode='constant'):
    """
    Silent version of create_patches_crop_then_pad - no print statements.
    """
    height, width = image.shape[:2]
    stride = int(patch_size * (1 - overlap_ratio))
    
    patches = []
    
    # Calculate patch positions (can extend beyond image boundaries)
    x_positions = list(range(0, width, stride))
    y_positions = list(range(0, height, stride))
    
    # Add final positions to ensure full coverage of image
    if x_positions[-1] + patch_size < width:
        x_positions.append(width - patch_size)
    if y_positions[-1] + patch_size < height:
        y_positions.append(height - patch_size)
    
    # Remove duplicates and sort
    x_positions = sorted(list(set(x_positions)))
    y_positions = sorted(list(set(y_positions)))
    
    # Extract patches using crop-then-pad
    for y in y_positions:
        for x in x_positions:
            # Extract patch with boundary-aware padding
            image_patch, mask_patch, padding_info = extract_patch_with_boundary_padding(
                image, mask, x, y, patch_size, padding_mode
            )
            
            # Check if this is a boundary patch
            is_boundary = (padding_info['top'] > 0 or padding_info['bottom'] > 0 or 
                          padding_info['left'] > 0 or padding_info['right'] > 0)
            
            # Create patch info
            patch_info = {
                'image_patch': image_patch,
                'mask_patch': mask_patch,
                'coordinates': (x, y),
                'patch_size': patch_size,
                'is_boundary_patch': is_boundary,
                'padding_info': padding_info,
                'strategy': 'crop_then_pad_silent'
            }
            
            patches.append(patch_info)
    
    return patches

def main():
    parser = argparse.ArgumentParser(
        description="Input, output, verbose arguments"
    )

    # Add arguments
    parser.add_argument("--input_im", "-i", help="Path to input images folder", default="segmentation/images_enhanced")
    parser.add_argument("--input_masks", "-m", help="Path to input masks folder", default="segmentation/masks")
    parser.add_argument("--output", "-o", help="Path to output file", default="coral_dataset_final_enhanced")
    parser.add_argument("--verbose", "-v",  action="store_true", help="Enable verbose mode")

    # Parse arguments
    args = parser.parse_args()

    process_segmentation_crop_then_pad_silent(
        images_dir=args.input_im,
        masks_dir=args.input_masks,
        output_dir=args.output,
        patch_size=1024,
        overlap_ratio=0.5,
        padding_mode='reflect',
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
#RUN python preprocessing.py -i segmentation/images_enhanced -m segmentation/masks
#ADD -v for verbose==True