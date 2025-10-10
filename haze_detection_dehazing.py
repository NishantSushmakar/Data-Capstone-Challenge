"""
Automated Haze Detection and Dehazing Techniques for Coral Reef Images

This module provides comprehensive haze detection and dehazing methods specifically
designed for underwater coral reef imagery. It can be integrated into the existing
preprocessing pipeline to improve image quality before cropping and padding.

Key Features:
- Multiple haze detection algorithms (statistical, learning-based, physics-based)
- Various dehazing techniques (DCP, CLAHE, Retinex, CNN-based)
- Automatic quality assessment and method selection
- Integration with existing coral preprocessing pipeline
- Batch processing capabilities
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import warnings
from tqdm import tqdm
import json
# from sklearn.metrics import structural_similarity as ssim
# from scipy import ndimage
# from skimage import exposure, filters, restoration
# from skimage.metrics import peak_signal_noise_ratio as psnr
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HazeDetector:
    """
    Automated haze detection using multiple complementary methods.
    """
    
    def __init__(self):
        self.methods = {
            'contrast_based': self._contrast_based_detection,
            'saturation_based': self._saturation_based_detection,
            'dark_channel': self._dark_channel_detection,
            'visibility_based': self._visibility_based_detection,
            'gradient_based': self._gradient_based_detection
        }
    
    def detect_haze(self, image: np.ndarray, method: str = 'ensemble') -> Dict:
        """
        Detect haze in an image using specified method or ensemble of methods.
        
        Args:
            image: Input RGB image (H, W, 3)
            method: Detection method ('contrast_based', 'saturation_based', 
                   'dark_channel', 'visibility_based', 'gradient_based', 'ensemble',
                   'custom_ensemble', 'contrast_visibility_ensemble')
        
        Returns:
            Dictionary with haze detection results
        """
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.astype(np.float32)
        
        if method == 'ensemble':
            return self._ensemble_detection(image_float)
        elif method == 'custom_ensemble':
            return self._custom_ensemble_detection(image_float)
        elif method == 'contrast_visibility_ensemble':
            return self._contrast_visibility_ensemble(image_float)
        elif method in self.methods:
            haze_score = self.methods[method](image_float)
            return {
                'method': method,
                'haze_score': haze_score,
                'has_haze': haze_score > 0.5,
                'confidence': abs(haze_score - 0.5) * 2
            }
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _contrast_based_detection(self, image: np.ndarray) -> float:
        """Detect haze based on image contrast."""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        contrast = gray.std() / gray.mean() if gray.mean() > 0 else 0
        
        # Normalize contrast score (lower contrast indicates more haze)
        # Typical clear images have contrast > 0.3, hazy images < 0.2
        haze_score = max(0, min(1, (0.4 - contrast) / 0.2))
        return haze_score
    
    def _saturation_based_detection(self, image: np.ndarray) -> float:
        """Detect haze based on color saturation."""
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        
        mean_saturation = np.mean(saturation)
        # Lower saturation indicates more haze
        haze_score = max(0, min(1, (0.6 - mean_saturation) / 0.4))
        return haze_score
    
    def _dark_channel_detection(self, image: np.ndarray) -> float:
        """Detect haze using Dark Channel Prior."""
        # Calculate dark channel
        dark_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(dark_channel, kernel)
        
        # Higher dark channel values indicate more haze
        mean_dark_channel = np.mean(dark_channel)
        haze_score = min(1, mean_dark_channel / 0.2)  # Normalize
        return haze_score
    
    def _visibility_based_detection(self, image: np.ndarray) -> float:
        """Detect haze based on visibility estimation."""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Calculate edge density as visibility measure
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Lower edge density indicates reduced visibility (more haze)
        haze_score = max(0, min(1, (0.1 - edge_density) / 0.08))
        return haze_score
    
    def _gradient_based_detection(self, image: np.ndarray) -> float:
        """Detect haze based on image gradients."""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        mean_gradient = np.mean(gradient_magnitude)
        # Lower gradients indicate more haze
        haze_score = max(0, min(1, (50 - mean_gradient) / 40))
        return haze_score
    
    def _ensemble_detection(self, image: np.ndarray) -> Dict:
        """Combine multiple detection methods for robust haze detection."""
        results = {}
        scores = []
        
        for method_name, method_func in self.methods.items():
            try:
                score = method_func(image)
                results[method_name] = score
                scores.append(score)
            except Exception as e:
                logger.warning(f"Method {method_name} failed: {e}")
                results[method_name] = 0.5  # Neutral score
                scores.append(0.5)
        
        # Weighted ensemble (some methods are more reliable for underwater images)
        weights = {
            'contrast_based': 0.25,
            'saturation_based': 0.3,  # Important for underwater images
            'dark_channel': 0.2,
            'visibility_based': 0.15,
            'gradient_based': 0.1
        }
        
        ensemble_score = sum(results[method] * weights[method] for method in results.keys())
        confidence = 1.0 - np.std(scores)  # Higher std = lower confidence
        
        return {
            'method': 'ensemble',
            'haze_score': ensemble_score,
            'has_haze': ensemble_score > 0.4,  # Lower threshold for underwater images
            'confidence': confidence,
            'individual_scores': results
        }
    
    def _contrast_visibility_ensemble(self, image: np.ndarray) -> Dict:
        """
        Custom ensemble using only contrast-based and visibility-based detection.
        Optimized for cases where you want to focus on these two specific methods.
        """
        results = {}
        scores = []
        
        # Only use contrast and visibility methods
        selected_methods = {
            'contrast_based': self._contrast_based_detection,
            'visibility_based': self._visibility_based_detection
        }
        
        for method_name, method_func in selected_methods.items():
            try:
                score = method_func(image)
                results[method_name] = score
                scores.append(score)
            except Exception as e:
                logger.warning(f"Method {method_name} failed: {e}")
                results[method_name] = 0.5  # Neutral score
                scores.append(0.5)
        
        # Equal weighting for the two methods
        weights = {
            'contrast_based': 0.6,
            'visibility_based': 0.4
        }
        
        ensemble_score = sum(results[method] * weights[method] for method in results.keys())
        confidence = 1.0 - np.std(scores) if len(scores) > 1 else 1.0
        
        return {
            'method': 'contrast_visibility_ensemble',
            'haze_score': ensemble_score,
            'has_haze': ensemble_score > 0.4,
            'confidence': confidence,
            'individual_scores': results,
            'weights_used': weights
        }
    
    def _custom_ensemble_detection(self, image: np.ndarray, 
                                 selected_methods: List[str] = None,
                                 custom_weights: Dict[str, float] = None) -> Dict:
        """
        Flexible custom ensemble allowing you to specify which methods to use and their weights.
        
        Args:
            image: Input image
            selected_methods: List of method names to include (default: ['contrast_based', 'visibility_based'])
            custom_weights: Dictionary of method weights (default: equal weights)
        """
        if selected_methods is None:
            selected_methods = ['contrast_based', 'visibility_based']
        
        if custom_weights is None:
            # Equal weights for selected methods
            weight_per_method = 1.0 / len(selected_methods)
            custom_weights = {method: weight_per_method for method in selected_methods}
        
        results = {}
        scores = []
        
        for method_name in selected_methods:
            if method_name in self.methods:
                try:
                    score = self.methods[method_name](image)
                    results[method_name] = score
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"Method {method_name} failed: {e}")
                    results[method_name] = 0.5  # Neutral score
                    scores.append(0.5)
            else:
                logger.warning(f"Unknown method: {method_name}")
        
        # Calculate weighted ensemble score
        ensemble_score = sum(results.get(method, 0.5) * custom_weights.get(method, 0) 
                           for method in selected_methods)
        confidence = 1.0 - np.std(scores) if len(scores) > 1 else 1.0
        
        return {
            'method': 'custom_ensemble',
            'haze_score': ensemble_score,
            'has_haze': ensemble_score > 0.4,
            'confidence': confidence,
            'individual_scores': results,
            'methods_used': selected_methods,
            'weights_used': custom_weights
        }


class ImageDehazer:
    """
    Multiple dehazing techniques for underwater coral reef images.
    """
    
    def __init__(self):
        self.methods = {
            'clahe': self._clahe_dehazing,
            'retinex': self._retinex_dehazing,
            # 'dark_channel_prior': self._dark_channel_prior_dehazing,
            # 'white_balance': self._white_balance_dehazing,
            # 'histogram_equalization': self._histogram_equalization_dehazing,
            # 'gamma_correction': self._gamma_correction_dehazing,
            'unsharp_masking': self._unsharp_masking_dehazing
        }
    
    def dehaze_image(self, image: np.ndarray, method: str = 'adaptive', 
                    haze_info: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Remove haze from image using specified method.
        
        Args:
            image: Input RGB image (H, W, 3)
            method: Dehazing method or 'adaptive' for automatic selection
            haze_info: Optional haze detection information for adaptive method
        
        Returns:
            Tuple of (dehazed_image, processing_info)
        """
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.astype(np.float32)
        
        if method == 'adaptive':
            return self._adaptive_dehazing(image_float, haze_info)
        elif method in self.methods:
            dehazed = self.methods[method](image_float)
            return dehazed, {'method': method, 'success': True}
        else:
            raise ValueError(f"Unknown dehazing method: {method}")
    
    def _clahe_dehazing(self, image: np.ndarray) -> np.ndarray:
        """Contrast Limited Adaptive Histogram Equalization."""
        # Convert to LAB color space for better results
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        dehazed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return dehazed.astype(np.float32) / 255.0
    def _retinex_dehazing(self, image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Controlled Retinex dehazing with blend factor alpha.
        Args:
            image: Input float32 image in [0,1].
            alpha: Retinex blend weight (0 = original only, 1 = full Retinex).
        Returns:
            Blended image in [0,1].
        """
        def single_scale_retinex(img_channel, sigma):
            # SSR: log-domain detail enhancement
            blur = cv2.GaussianBlur(img_channel, (0, 0), sigma)
            return np.log(img_channel + 1e-6) - np.log(blur + 1e-6)
        
        # Single scale to limit color distortion
        sigma = 15
        retinex_result = np.zeros_like(image)
        
        # Compute Retinex per channel
        for c in range(3):
            retinex_result[:, :, c] = single_scale_retinex(image[:, :, c], sigma)
        
        # Normalize to [0,1]
        for c in range(3):
            channel = retinex_result[:, :, c]
            retinex_result[:, :, c] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-6)
        
        # Blend with original
        blended = alpha * retinex_result + (1 - alpha) * image
        return np.clip(blended, 0, 1)


    # def _retinex_dehazing(self, image: np.ndarray, alpha=0.7) -> np.ndarray:
    #     """Multi-Scale Retinex for haze removal."""
    #     def single_scale_retinex(img, sigma):
    #         retinex = np.log10(img + 1e-6) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1e-6)
    #         return alpha * retinex + (1 - alpha) * image
        
        # # Apply multi-scale retinex
        # scales = [15, 80, 250]
        # retinex_sum = np.zeros_like(image)
        
        # for scale in scales:
        #     for i in range(3):  # Process each channel
        #         retinex_sum[:, :, i] += single_scale_retinex(image[:, :, i], scale)
        
        # retinex_sum /= len(scales)
        
        # # Normalize to [0, 1]
        # for i in range(3):
        #     channel = retinex_sum[:, :, i]
        #     channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-6)
        #     retinex_sum[:, :, i] = channel
        
        # return retinex_sum
    
    def _dark_channel_prior_dehazing(self, image: np.ndarray) -> np.ndarray:
        """Dark Channel Prior based dehazing."""
        def get_dark_channel(img, size=15):
            dark = np.min(img, axis=2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            dark = cv2.erode(dark, kernel)
            return dark
        
        def get_atmosphere(img, dark_channel, percent=0.1):
            h, w = dark_channel.shape
            num_pixels = int(h * w * percent)
            dark_flat = dark_channel.flatten()
            indices = np.argpartition(dark_flat, -num_pixels)[-num_pixels:]
            
            atmosphere = np.zeros(3)
            for idx in indices:
                y, x = divmod(idx, w)
                atmosphere = np.maximum(atmosphere, img[y, x])
            
            return atmosphere
        
        def get_transmission(img, atmosphere, omega=0.95, size=15):
            transmission = 1 - omega * get_dark_channel(img / atmosphere, size)
            return transmission
        
        # Estimate atmospheric light
        dark_channel = get_dark_channel(image)
        atmosphere = get_atmosphere(image, dark_channel)
        
        # Estimate transmission map
        transmission = get_transmission(image, atmosphere)
        transmission = np.maximum(transmission, 0.1)  # Avoid division by zero
        
        # Recover scene radiance
        dehazed = np.zeros_like(image)
        for i in range(3):
            dehazed[:, :, i] = (image[:, :, i] - atmosphere[i]) / transmission + atmosphere[i]
        
        return np.clip(dehazed, 0, 1)
    
    def _white_balance_dehazing(self, image: np.ndarray) -> np.ndarray:
        """White balance correction for underwater images."""
        # Calculate channel means
        means = np.mean(image.reshape(-1, 3), axis=0)
        
        # Use gray world assumption
        gray_world = np.mean(means)
        
        # Calculate correction factors
        correction = gray_world / (means + 1e-6)
        
        # Apply correction
        dehazed = image * correction[np.newaxis, np.newaxis, :]
        return np.clip(dehazed, 0, 1)
    
    def _histogram_equalization_dehazing(self, image: np.ndarray) -> np.ndarray:
        """Histogram equalization for each channel."""
        dehazed = np.zeros_like(image)
        
        for i in range(3):
            channel = (image[:, :, i] * 255).astype(np.uint8)
            equalized = cv2.equalizeHist(channel)
            dehazed[:, :, i] = equalized.astype(np.float32) / 255.0
        
        return dehazed
    
    def _gamma_correction_dehazing(self, image: np.ndarray, gamma: float = 0.7) -> np.ndarray:
        """Gamma correction to brighten dark regions."""
        return np.power(image, gamma)
    
    def _unsharp_masking_dehazing(self, image: np.ndarray) -> np.ndarray:
        """Unsharp masking to enhance details."""
        # Convert to uint8 for OpenCV
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(img_uint8, (0, 0), 2.0)
        
        # Unsharp masking
        sharpened = cv2.addWeighted(img_uint8, 1.5, blurred, -0.5, 0)
        
        return sharpened.astype(np.float32) / 255.0
    def _guided_filter(self, image: np.ndarray, radius: int = 4, eps: float = 0.01) -> np.ndarray:
        """
        Guided filter enhancement for edge-preserving smoothing.
        Uses the image as its own guide.
        """
        def box_filter(img, r):
            kernel_size = 2 * r + 1
            return cv2.blur(img, (kernel_size, kernel_size))
        
        # Ensure float32
        if image.dtype != np.float32:
            img = image.astype(np.float32)
        else:
            img = image.copy()
        
        H, W = img.shape[:2]
        N = box_filter(np.ones((H, W)), radius)
        
        # Process each channel
        output_channels = []
        
        for c in range(img.shape[2]):
            I = img[:, :, c]  # Guide (same as input)
            p = img[:, :, c]  # Input
            
            # Compute local statistics
            mean_I = box_filter(I, radius) / N
            mean_p = box_filter(p, radius) / N
            mean_Ip = box_filter(I * p, radius) / N
            mean_II = box_filter(I * I, radius) / N
            
            # Compute coefficients
            var_I = mean_II - mean_I * mean_I
            cov_Ip = mean_Ip - mean_I * mean_p
            
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            
            # Smooth coefficients
            mean_a = box_filter(a, radius) / N
            mean_b = box_filter(b, radius) / N
            
            # Output
            q = mean_a * I + mean_b
            output_channels.append(q)
        
        # Stack channels
        result = np.stack(output_channels, axis=2)
        return np.clip(result, 0, 1)

    def _adaptive_dehazing(self, image: np.ndarray, haze_info: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Adaptive dehazing optimized for contrast and visibility enhancement.
        Prioritizes CLAHE and Retinex methods, with combined approach for best results.
        """
        if haze_info is None:
            detector = HazeDetector()
            haze_info = detector.detect_haze(image)
        
        haze_score = haze_info.get('haze_score', 0.5)

        if haze_score < 0.3:

            method='clahe'
            dehazed=self._clahe_dehazing(image)
        elif haze_score < 0.5:
            method='clahe_guided'
            clahe_result = self._clahe_dehazing(image)
            dehazed = self._guided_filter(clahe_result)
        else:
            method='retinex_clahe'
            # Gentle retinex with strong color preservation
            retinex_result = self._retinex_dehazing(image, alpha=0.4)  # Blend with original
            dehazed = self._clahe_dehazing(retinex_result)
        # # New adaptive strategy focused on contrast and visibility
        # if haze_score < 0.3:
        #     # Light haze - still apply CLAHE for contrast enhancement
        #     # Even "clear" underwater images benefit from contrast improvement
        #     method = 'clahe'
        #     dehazed = self.methods[method](image)
            
        # elif haze_score < 0.6:
        #     # Moderate haze - combine Retinex + CLAHE for best of both
        #     method = 'retinex_clahe_combined'
            
        #     # Apply Retinex first for global illumination correction
        #     retinex_result = self._retinex_dehazing(image)
            
        #     # Then apply CLAHE for local contrast enhancement
        #     clahe_result = self._clahe_dehazing(retinex_result)
            
        #     dehazed = clahe_result
            
        # else:
        #     # Heavy haze - use Retinex first, then CLAHE, with stronger processing
        #     method = 'retinex_clahe_heavy'
            
        #     # Apply Retinex with stronger effect
        #     retinex_result = self._retinex_dehazing(image)
            
        #     # Apply CLAHE with stronger parameters
        #     clahe_result = self._clahe_dehazing(retinex_result)
            
        #     # Optional: slight unsharp masking for visibility enhancement
        #     unsharp_result = self._unsharp_masking_dehazing(clahe_result)
            
        #     # Blend CLAHE and unsharp results (favor CLAHE for contrast focus)
        #     dehazed = 0.8 * clahe_result + 0.2 * unsharp_result
        
        processing_info = {
            'method': f'adaptive_{method}',
            'haze_score': haze_score,
            'success': True,
            'techniques_used': self._get_techniques_used(method),
            'optimized_for': 'contrast_and_visibility'
        }
        
        return dehazed, processing_info
    
    def _get_techniques_used(self, method: str) -> List[str]:
        """Helper method to track which techniques were used."""
        if method == 'clahe':
            return ['CLAHE']
        elif method == 'clahe_guided':
            return ['Multi-Scale Retinex', 'CLAHE']
        elif method == 'clahe_retinex':
            return ['Multi-Scale Retinex', 'CLAHE', 'Unsharp Masking']
        else:
            return [method]


class ImageQualityAssessor:
    """
    Assess image quality before and after dehazing.
    """
    
    def assess_quality(self, image: np.ndarray) -> Dict:
        """
        Comprehensive image quality assessment.
        
        Args:
            image: Input RGB image (H, W, 3)
        
        Returns:
            Dictionary with quality metrics
        """
        if image.dtype != np.uint8:
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image_uint8 = image
        
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        
        metrics = {
            'contrast': self._calculate_contrast(gray),
            'sharpness': self._calculate_sharpness(gray),
            'brightness': self._calculate_brightness(gray),
            'saturation': self._calculate_saturation(image_uint8),
            'entropy': self._calculate_entropy(gray),
            'edge_density': self._calculate_edge_density(gray)
        }
        
        # Overall quality score (0-1, higher is better)
        weights = {
            'contrast': 0.2,
            'sharpness': 0.25,
            'brightness': 0.1,
            'saturation': 0.2,
            'entropy': 0.15,
            'edge_density': 0.1
        }
        
        # Normalize metrics to 0-1 scale
        normalized_metrics = self._normalize_metrics(metrics)
        
        overall_score = sum(normalized_metrics[metric] * weights[metric] 
                          for metric in weights.keys())
        
        metrics['overall_quality'] = overall_score
        metrics['normalized_metrics'] = normalized_metrics
        
        return metrics
    
    def _calculate_contrast(self, gray: np.ndarray) -> float:
        """Calculate RMS contrast."""
        return gray.std()
    
    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Calculate sharpness using Laplacian variance."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    def _calculate_brightness(self, gray: np.ndarray) -> float:
        """Calculate average brightness."""
        return gray.mean()
    
    def _calculate_saturation(self, image: np.ndarray) -> float:
        """Calculate average saturation."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return hsv[:, :, 1].mean()
    
    def _calculate_entropy(self, gray: np.ndarray) -> float:
        """Calculate image entropy."""
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        """Calculate edge density."""
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / edges.size
    
    def _normalize_metrics(self, metrics: Dict) -> Dict:
        """Normalize metrics to 0-1 scale."""
        # Typical ranges for normalization
        ranges = {
            'contrast': (0, 100),
            'sharpness': (0, 1000),
            'brightness': (0, 255),
            'saturation': (0, 255),
            'entropy': (0, 8),
            'edge_density': (0, 0.2)
        }
        
        normalized = {}
        for metric, value in metrics.items():
            if metric in ranges:
                min_val, max_val = ranges[metric]
                normalized[metric] = np.clip((value - min_val) / (max_val - min_val), 0, 1)
        
        return normalized


class CoralImageEnhancer:
    """
    Main class for coral image enhancement with haze detection and dehazing.
    """
    
    def __init__(self):
        self.haze_detector = HazeDetector()
        self.dehazer = ImageDehazer()
        self.quality_assessor = ImageQualityAssessor()
    
    def enhance_image(self, image: np.ndarray, 
                     detection_method: str = 'ensemble',
                     dehazing_method: str = 'adaptive',
                     quality_threshold: float = 0.3) -> Tuple[np.ndarray, Dict]:
        """
        Complete image enhancement pipeline.
        
        Args:
            image: Input RGB image (H, W, 3)
            detection_method: Haze detection method
            dehazing_method: Dehazing method
            quality_threshold: Minimum quality improvement threshold
        
        Returns:
            Tuple of (enhanced_image, enhancement_info)
        """
        # Step 1: Assess original image quality
        original_quality = self.quality_assessor.assess_quality(image)
        
        # Step 2: Detect haze
        haze_info = self.haze_detector.detect_haze(image, detection_method)
        
        # Step 3: Apply dehazing if haze is detected
        if haze_info['has_haze'] and haze_info['confidence'] > 0.3:
            enhanced_image, dehazing_info = self.dehazer.dehaze_image(
                image, dehazing_method, haze_info
            )
            
            # Step 4: Assess enhanced image quality
            enhanced_quality = self.quality_assessor.assess_quality(enhanced_image)
            
            # Step 5: Decide whether to keep enhancement
            quality_improvement = (enhanced_quality['overall_quality'] - 
                                 original_quality['overall_quality'])
            
            if quality_improvement > quality_threshold:
                final_image = enhanced_image
                enhancement_applied = True
            else:
                final_image = image
                enhancement_applied = False
        else:
            final_image = image
            enhanced_quality = original_quality
            dehazing_info = {'method': 'none', 'reason': 'no_haze_detected'}
            enhancement_applied = False
            quality_improvement = 0
        
        # Compile enhancement information
        enhancement_info = {
            'haze_detection': haze_info,
            'dehazing': dehazing_info,
            'original_quality': original_quality,
            'enhanced_quality': enhanced_quality,
            'quality_improvement': quality_improvement,
            'enhancement_applied': enhancement_applied
        }
        
        return final_image, enhancement_info
    
    def process_image_file(self, image_path: Union[str, Path], 
                          output_path: Optional[Union[str, Path]] = None,
                          **kwargs) -> Dict:
        """
        Process a single image file.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save enhanced image
            **kwargs: Additional arguments for enhance_image
        
        Returns:
            Processing results dictionary
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Enhance image
            enhanced_image, enhancement_info = self.enhance_image(image, **kwargs)
            
            # Save enhanced image if output path provided
            if output_path is not None:
                enhanced_bgr = cv2.cvtColor(
                    (enhanced_image * 255).astype(np.uint8), 
                    cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(str(output_path), enhanced_bgr)
            
            return {
                'input_path': str(image_path),
                'output_path': str(output_path) if output_path else None,
                'success': True,
                'enhancement_info': enhancement_info
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'input_path': str(image_path),
                'output_path': None,
                'success': False,
                'error': str(e)
            }
    
    def batch_process(self, input_dir: Union[str, Path], 
                     output_dir: Union[str, Path],
                     file_pattern: str = "*.jpg",
                     **kwargs) -> Dict:
        """
        Batch process images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save enhanced images
            file_pattern: File pattern to match (e.g., "*.jpg", "*.png")
            **kwargs: Additional arguments for enhance_image
        
        Returns:
            Batch processing results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all matching images
        image_files = list(input_path.glob(file_pattern))
        
        if not image_files:
            logger.warning(f"No images found matching pattern {file_pattern} in {input_dir}")
            return {'success': False, 'error': 'No images found'}
        
        results = []
        successful = 0
        failed = 0
        
        logger.info(f"Processing {len(image_files)} images...")
        
        for image_file in tqdm(image_files, desc="Enhancing images"):
            output_file = output_path / image_file.name
            
            result = self.process_image_file(
                image_file, output_file, **kwargs
            )
            
            results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed += 1
        
        # Compile batch statistics
        batch_results = {
            'total_images': len(image_files),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(image_files),
            'results': results,
            'input_directory': str(input_dir),
            'output_directory': str(output_dir)
        }
        
        logger.info(f"Batch processing complete: {successful}/{len(image_files)} successful")
        
        return batch_results


def demonstrate_haze_detection_dehazing():
    """
    Demonstration function showing how to use the haze detection and dehazing system.
    """
    print("🌊 Coral Image Haze Detection and Dehazing Demo")
    print("=" * 60)
    
    # Initialize enhancer
    enhancer = CoralImageEnhancer()
    
    # Example usage for single image
    print("\n📸 Single Image Enhancement:")
    print("enhancer = CoralImageEnhancer()")
    print("enhanced_image, info = enhancer.enhance_image(image)")
    print("print(f'Quality improvement: {info[\"quality_improvement\"]:.3f}')")
    
    # Example usage for batch processing
    print("\n📁 Batch Processing:")
    print("results = enhancer.batch_process(")
    print("    input_dir='segmentation/images',")
    print("    output_dir='enhanced_images',")
    print("    detection_method='ensemble',")
    print("    dehazing_method='adaptive'")
    print(")")
    
    # Available methods
    print("\n🔧 Available Methods:")
    print("Haze Detection Methods:")
    for method in enhancer.haze_detector.methods.keys():
        print(f"  - {method}")
    print("  - ensemble (recommended)")
    
    print("\nDehazing Methods:")
    for method in enhancer.dehazer.methods.keys():
        print(f"  - {method}")
    print("  - adaptive (recommended)")
    
    print("\n💡 Integration with Existing Pipeline:")
    print("# Add this before your existing crop_then_pad processing:")
    print("enhancer = CoralImageEnhancer()")
    print("enhanced_image, _ = enhancer.enhance_image(original_image)")
    print("# Then use enhanced_image in your existing pipeline")


if __name__ == "__main__":
    demonstrate_haze_detection_dehazing()


