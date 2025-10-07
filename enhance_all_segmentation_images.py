#!/usr/bin/env python3
"""
Enhance All Segmentation Images

This script processes all images in segmentation/images/ and saves enhanced versions
in segmentation/images_enhanced/ using your custom settings:
- Contrast+Visibility ensemble (60/40 weights)  
- CLAHE+Retinex adaptive enhancement
- Uses existing 0.4 haze threshold logic in the pipeline
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import shutil
import argparse

# Add current directory to path
# sys.path.append(str(Path(__file__).parent))

try:
    from haze_detection_dehazing import CoralImageEnhancer, HazeDetector
except ImportError as e:
    print(f" Import error: {e}")
    sys.exit(1)


def enhance_all_segmentation_images(
    input_dir='segmentation/images',
    output_dir='segmentation/images_enhanced',
    verbose=False
):
    """
    Process all images in segmentation/images and save enhanced versions.
    
    Args:
        input_dir: Input directory with original images
        output_dir: Output directory for enhanced images
        verbose: Show detailed progress
    
    Returns:
        Dictionary with processing statistics
    """
    
    if verbose:
        print(f" ENHANCING ALL SEGMENTATION IMAGES")
        print("=" * 60)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Detection: Contrast+Visibility ensemble (60/40 weights)")
        print(f"Enhancement: CLAHE+Retinex adaptive method")
        print(f"Haze threshold: Built-in 0.4 threshold logic")
    
    # Check input directory
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f" Input directory not found: {input_dir}")
        return None
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = list(input_path.glob("*.jpg"))
    if not image_files:
        print(f" No .jpg images found in {input_dir}")
        return None
    
    if verbose:
        print(f"\\nFound {len(image_files)} images to process")
    
    # Initialize enhancer
    enhancer = CoralImageEnhancer()
    
    # Statistics tracking
    stats = {
        'total_images': len(image_files),
        'processed_successfully': 0,
        'enhanced_images': 0,
        'skipped_images': 0,
        'failed_images': 0,
        'haze_detected_count': 0,
        'enhancement_methods_used': {},
        'haze_scores': [],
        'quality_improvements': [],
        'errors': []
    }
    
    # Process each image
    if verbose:
        print(f"\\n Processing images...")
        progress_bar = tqdm(image_files, desc="Enhancing")
    else:
        progress_bar = image_files
    
    for image_path in progress_bar:
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                stats['errors'].append(f"Could not load {image_path.name}")
                stats['failed_images'] += 1
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply enhancement with your settings
            enhanced_image, enhancement_info = enhancer.enhance_image(
                image,
                detection_method='contrast_visibility_ensemble',  # Your 60/40 weights
                dehazing_method='adaptive',                      # Your CLAHE+Retinex method
                quality_threshold=0.1  # Lower threshold since we want to see results
            )
            
            # Track statistics
            haze_score = enhancement_info['haze_detection']['haze_score']
            stats['haze_scores'].append(haze_score)
            
            if enhancement_info['haze_detection']['has_haze']:
                stats['haze_detected_count'] += 1
            
            if enhancement_info['enhancement_applied']:
                stats['enhanced_images'] += 1
                stats['quality_improvements'].append(enhancement_info['quality_improvement'])
                
                # Track methods used
                method = enhancement_info['dehazing']['method']
                if method in stats['enhancement_methods_used']:
                    stats['enhancement_methods_used'][method] += 1
                else:
                    stats['enhancement_methods_used'][method] = 1
                
                # Save enhanced image
                output_filename = output_path / image_path.name
                enhanced_bgr = cv2.cvtColor((enhanced_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_filename), enhanced_bgr)
                
            else:
                stats['skipped_images'] += 1
                # Optionally save original image (uncomment if you want this)
                output_filename = output_path / image_path.name
                shutil.copy2(image_path, output_filename)
            
            stats['processed_successfully'] += 1
            
        except Exception as e:
            error_msg = f"Error processing {image_path.name}: {str(e)}"
            stats['errors'].append(error_msg)
            stats['failed_images'] += 1
            if verbose:
                print(f"\\n  {error_msg}")
    
    if verbose and hasattr(progress_bar, 'close'):
        progress_bar.close()
    
    # Calculate final statistics
    if stats['quality_improvements']:
        stats['average_quality_improvement'] = np.mean(stats['quality_improvements'])
    else:
        stats['average_quality_improvement'] = 0.0
    
    if stats['haze_scores']:
        stats['haze_score_stats'] = {
            'mean': np.mean(stats['haze_scores']),
            'std': np.std(stats['haze_scores']),
            'min': np.min(stats['haze_scores']),
            'max': np.max(stats['haze_scores'])
        }
    
    # Save statistics
    stats_file = output_path / 'enhancement_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Display results
    if verbose:
        print(f"\\n PROCESSING COMPLETE!")
        print("=" * 50)
        print(f" RESULTS:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Processed successfully: {stats['processed_successfully']}")
        print(f"  Enhanced images: {stats['enhanced_images']}")
        print(f"  Skipped images: {stats['skipped_images']}")
        print(f"  Failed images: {stats['failed_images']}")
        
        print(f"\\n HAZE DETECTION:")
        print(f"  Images with haze detected: {stats['haze_detected_count']}/{stats['total_images']} ({stats['haze_detected_count']/stats['total_images']*100:.1f}%)")
        
        if stats['haze_scores']:
            haze_stats = stats['haze_score_stats']
            print(f"  Haze score range: {haze_stats['min']:.3f} - {haze_stats['max']:.3f}")
            print(f"  Average haze score: {haze_stats['mean']:.3f} ± {haze_stats['std']:.3f}")
        
        print(f"\\n ENHANCEMENT RESULTS:")
        print(f"  Enhancement rate: {stats['enhanced_images']}/{stats['total_images']} ({stats['enhanced_images']/stats['total_images']*100:.1f}%)")
        
        if stats['quality_improvements']:
            print(f"  Average quality improvement: {stats['average_quality_improvement']:.3f}")
        
        if stats['enhancement_methods_used']:
            print(f"  Methods used:")
            for method, count in stats['enhancement_methods_used'].items():
                print(f"    {method}: {count} images")
        
        print(f"\\n OUTPUT:")
        print(f"  Enhanced images saved to: {output_dir}")
        print(f"  Statistics saved to: {stats_file}")
        
        if stats['errors']:
            print(f"\\n ERRORS ({len(stats['errors'])}):")
            for error in stats['errors'][:5]:  # Show first 5 errors
                print(f"    {error}")
            if len(stats['errors']) > 5:
                print(f"    ... and {len(stats['errors']) - 5} more errors")
    
    return stats


def main():
    """
    Main function to enhance all segmentation images.
    """
    parser = argparse.ArgumentParser(
        description="Input, output, verbose arguments"
    )

    # Add arguments
    parser.add_argument("--input_im", "-i", help="Path to input images folder", default="segmentation/images")
    parser.add_argument("--output", "-o", help="Path to output folder", default="segmentation/images_enhanced")
    parser.add_argument("--verbose", "-v",  action="store_true", help="Enable verbose mode")

    # Parse arguments
    args = parser.parse_args()
    
    # Check if segmentation directory exists
    if not Path(args.input_im).exists():
        print(f" {args.input_im} directory not found")
        print("Please ensure you're running this from the correct directory")
        return False
    
    try:
        # Process all images
        stats = enhance_all_segmentation_images(
            input_dir=args.input_im,
            output_dir=args.output,
            verbose=args.verbose
        )
        
        if stats:
            print("\\n Enhancement completed successfully!")
            print(f"Enhanced {stats['enhanced_images']} out of {stats['total_images']} images")
            return True
        else:
            print("\\n Enhancement failed")
            return False
            
    except Exception as e:
        print(f"\\n Error during processing: {e}")
        return False


if __name__ == "__main__":
    main()
    
#RUN python enhance_segmentation_images.py -i segmentation/images -o segmentation/images_enhanced
#ADD -v for verbose==TRUE