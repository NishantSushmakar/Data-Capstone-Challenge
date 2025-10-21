"""
Improved Coral Bleaching Detection Model
Key improvements:
1. Stronger backbone (EfficientNet-B2) with attention mechanism
2. ASPP module for multi-scale features
3. Advanced loss functions (Focal + Dice)
4. Enhanced data augmentation for underwater images
5. Multi-task learning (segmentation + bleaching regression)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from typing import Tuple, Dict, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from scipy.ndimage import distance_transform_edt

# ======================== Enhanced Dataset ========================

class ImprovedCoralDataset(Dataset):
    """Enhanced dataset with boundary detection and data quality filtering"""
    
    def __init__(self, 
                 image_dir: str,
                 bleached_mask_dir: str,
                 non_bleached_mask_dir: str,
                 image_ids: List[str],
                 transform=None,
                 mode='train',
                 filter_low_quality=True):
        self.image_dir = image_dir
        self.bleached_mask_dir = bleached_mask_dir
        self.non_bleached_mask_dir = non_bleached_mask_dir
        self.image_ids = image_ids
        self.transform = transform
        self.mode = mode
        
        # Filter out low-quality samples (optional)
        if filter_low_quality and mode == 'train':
            self.image_ids = self._filter_quality(image_ids)
        
    def _filter_quality(self, image_ids):
        """Filter out samples with too little coral coverage"""
        filtered_ids = []
        for img_id in image_ids:
            bleached_mask_path = os.path.join(self.bleached_mask_dir, f"{img_id}_bleached.png")
            non_bleached_mask_path = os.path.join(self.non_bleached_mask_dir, f"{img_id}_non_bleached.png")
            
            bleached_mask = np.array(Image.open(bleached_mask_path).convert('L'))
            non_bleached_mask = np.array(Image.open(non_bleached_mask_path).convert('L'))
            
            total_coral = ((bleached_mask > 127).sum() + (non_bleached_mask > 127).sum())
            coverage = total_coral / (bleached_mask.shape[0] * bleached_mask.shape[1])
            
            # Keep samples with at least 5% coral coverage
            if coverage > 0.05:
                filtered_ids.append(img_id)
        
        print(f"Filtered dataset: {len(filtered_ids)}/{len(image_ids)} samples retained")
        return filtered_ids
    
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Load image and masks
        image_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        bleached_mask_path = os.path.join(self.bleached_mask_dir, f"{img_id}_bleached.png")
        non_bleached_mask_path = os.path.join(self.non_bleached_mask_dir, f"{img_id}_non_bleached.png")
        
        image = np.array(Image.open(image_path).convert('RGB'))
        bleached_mask = np.array(Image.open(bleached_mask_path).convert('L'))
        non_bleached_mask = np.array(Image.open(non_bleached_mask_path).convert('L'))
        
        # Convert to binary
        bleached_mask = (bleached_mask > 127).astype(np.float32)
        non_bleached_mask = (non_bleached_mask > 127).astype(np.float32)
        
        # Create segmentation mask
        segmentation_mask = np.zeros_like(bleached_mask, dtype=np.int64)
        segmentation_mask[non_bleached_mask == 1] = 1
        segmentation_mask[bleached_mask == 1] = 2
        
        # Calculate metrics
        total_coral_pixels = (bleached_mask == 1).sum() + (non_bleached_mask == 1).sum()
        if total_coral_pixels > 0:
            bleaching_ratio = (bleached_mask == 1).sum() / total_coral_pixels
            coral_coverage = total_coral_pixels / (bleached_mask.shape[0] * bleached_mask.shape[1])
        else:
            bleaching_ratio = 0.0
            coral_coverage = 0.0
        
        # Data augmentation
        if self.transform:
            transformed = self.transform(image=image, mask=segmentation_mask)
            image = transformed['image']
            segmentation_mask = transformed['mask']
        
        # Convert to tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(segmentation_mask, torch.Tensor):
            segmentation_mask = torch.from_numpy(segmentation_mask).long()
        else:
            segmentation_mask = segmentation_mask.long()

        return {
            'image': image,
            'segmentation_mask': segmentation_mask,
            'bleaching_ratio': torch.tensor(bleaching_ratio, dtype=torch.float32),
            'coral_coverage': torch.tensor(coral_coverage, dtype=torch.float32),
            'image_id': img_id
        }

# ======================== Attention Modules ========================

class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ======================== ASPP Module ========================

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=False)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.conv_out(x)

# ======================== Improved Model ========================

class ImprovedCoralModel(nn.Module):
    """Improved coral bleaching detection model"""
    
    def __init__(self, 
                 backbone_name='tf_efficientnet_b3_ns',  # Upgraded backbone
                 num_classes=3,
                 pretrained=True):
        super().__init__()
        
        # Stronger backbone with noisy student weights
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4]
        )
        
        channels = self.backbone.feature_info.channels()
        
        # ASPP module for multi-scale features
        self.aspp = ASPP(channels[3], 256)
        
        # Decoder with attention
        self.decoder4 = self._make_decoder_block(256, channels[2], 256)
        self.cbam4 = CBAM(256)
        
        self.decoder3 = self._make_decoder_block(256 + channels[2], channels[1], 128)
        self.cbam3 = CBAM(128)
        
        self.decoder2 = self._make_decoder_block(128 + channels[1], channels[0], 64)
        self.cbam2 = CBAM(64)
        
        self.decoder1 = self._make_decoder_block(64 + channels[0], channels[0], 32)
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Enhanced regression heads with more capacity
        self.bleaching_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.coverage_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def _make_decoder_block(self, in_channels, skip_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        features = self.backbone(x)
        
        # ASPP on deepest features
        x_aspp = self.aspp(features[3])
        
        # Decoder with attention
        d4 = self.decoder4(x_aspp)
        d4 = self.cbam4(d4)
        d4_up = F.interpolate(d4, size=features[2].shape[2:], mode='bilinear', align_corners=False)
        
        d3 = self.decoder3(torch.cat([d4_up, features[2]], dim=1))
        d3 = self.cbam3(d3)
        d3_up = F.interpolate(d3, size=features[1].shape[2:], mode='bilinear', align_corners=False)
        
        d2 = self.decoder2(torch.cat([d3_up, features[1]], dim=1))
        d2 = self.cbam2(d2)
        d2_up = F.interpolate(d2, size=features[0].shape[2:], mode='bilinear', align_corners=False)
        
        d1 = self.decoder1(torch.cat([d2_up, features[0]], dim=1))
        
        # Outputs
        seg_logits = self.seg_head(d1)
        seg_output = F.interpolate(seg_logits, size=input_size, mode='bilinear', align_corners=False)

        # Global features
        global_features = self.global_pool(x_aspp).flatten(1)

        bleaching_ratio = self.bleaching_head(global_features)
        coral_coverage = self.coverage_head(global_features)

        return {
            'segmentation': seg_output,
            'bleaching_ratio': bleaching_ratio.squeeze(1),
            'coral_coverage': coral_coverage.squeeze(1)
        }

# ======================== Advanced Loss Functions ========================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss for better boundary prediction"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets, num_classes=3):
        inputs = F.softmax(inputs, dim=1)
        
        dice_loss = 0
        for cls in range(num_classes):
            input_cls = inputs[:, cls, :, :]
            target_cls = (targets == cls).float()
            
            intersection = (input_cls * target_cls).sum(dim=(1, 2))
            union = input_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice).mean()
        
        return dice_loss / num_classes

class CombinedLoss(nn.Module):
    """Combined loss WITHOUT boundary component"""
    def __init__(self,
                 seg_weight=2.0,
                 dice_weight=1.5,
                 bleaching_weight=2.0,
                 coverage_weight=0.8):
        super().__init__()
        self.seg_weight = seg_weight
        self.dice_weight = dice_weight
        self.bleaching_weight = bleaching_weight
        self.coverage_weight = coverage_weight

        # Focal loss with class weights
        class_weights = torch.tensor([0.5, 1.0, 1.5])  # background, healthy, bleached
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=2)
        self.dice_loss = DiceLoss()
        self.huber_loss = nn.HuberLoss(delta=0.1)

    def forward(self, outputs, targets):
        # Segmentation losses
        focal = self.focal_loss(outputs['segmentation'], targets['segmentation_mask'])
        dice = self.dice_loss(outputs['segmentation'], targets['segmentation_mask'])
        seg_loss = self.seg_weight * focal + self.dice_weight * dice

        # Regression losses (use Huber for robustness)
        bleaching_loss = self.bleaching_weight * self.huber_loss(
            outputs['bleaching_ratio'],
            targets['bleaching_ratio']
        )

        coverage_loss = self.coverage_weight * self.huber_loss(
            outputs['coral_coverage'],
            targets['coral_coverage']
        )

        total_loss = seg_loss + bleaching_loss + coverage_loss

        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'bleaching_loss': bleaching_loss,
            'coverage_loss': coverage_loss
        }

# ======================== Enhanced Data Augmentation ========================

def get_enhanced_transforms(mode='train', img_size=512):
    """Enhanced data augmentation for underwater images"""
    
    if mode == 'train':
        return A.Compose([
            # Geometric transforms
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-30, 30), p=0.5),
            
            # Color augmentation (important for underwater images)
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
                A.RandomGamma(gamma_limit=(70, 130), p=1),
                A.CLAHE(clip_limit=4.0, p=1),  # Enhance underwater contrast
            ], p=0.9),
            
            # Simulate underwater conditions
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0), p=1),  # Fixed for albumentations 2.0
                A.GaussianBlur(blur_limit=(3, 5), p=1),
                A.MedianBlur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=5, p=1),
            ], p=0.4),

            # Simulate color cast (common in underwater images)
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

            # Occlusion augmentation for robustness (albumentations 2.0 syntax)
            A.CoarseDropout(
                num_holes_range=(5, 10),
                hole_height_range=(32, 64),
                hole_width_range=(32, 64),
                fill=0,
                p=0.3
            ),

            # Normalize
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'masks': 'masks'})
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'masks': 'masks'})

# ======================== Metrics ========================

class MetricCalculator:
    @staticmethod
    def calculate_iou(pred_seg, true_seg, num_classes=3):
        """Calculate mean IoU"""
        pred_seg = pred_seg.argmax(dim=1)
        ious = []
        
        for cls in range(1, num_classes):  # Skip background
            pred_cls = (pred_seg == cls)
            true_cls = (true_seg == cls)
            
            intersection = (pred_cls & true_cls).float().sum()
            union = (pred_cls | true_cls).float().sum()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou.item())
        
        return np.mean(ious) if ious else 0.0
    
    @staticmethod
    def calculate_boundary_f1(pred_boundary, true_boundary, threshold=0.5):
        """Calculate boundary F1 score with relaxed matching"""
        pred_binary = (pred_boundary > threshold).float()

        # If no boundary pixels in ground truth, return 0
        if true_boundary.sum() < 1:
            return 0.0

        # Relax matching: dilate predicted boundary for tolerance
        # Convert to numpy for morphological operations
        pred_np = pred_binary.squeeze().cpu().numpy()
        true_np = true_boundary.squeeze().cpu().numpy()

        # Dilate prediction slightly to allow for small misalignments
        kernel = np.ones((3, 3), np.uint8)
        pred_dilated = cv2.dilate(pred_np.astype(np.uint8), kernel, iterations=1)
        pred_dilated = torch.from_numpy(pred_dilated).float().to(pred_boundary.device)

        tp = (pred_dilated * true_boundary).sum()
        fp = (pred_binary * (1 - true_boundary)).sum()
        fn = ((1 - pred_dilated) * true_boundary).sum()

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        return f1.item()
    
    @staticmethod
    def calculate_mae(pred, true):
        """Mean Absolute Error"""
        return torch.abs(pred - true).mean().item()
    
    @staticmethod
    def calculate_accuracy_threshold(pred, true, threshold=0.1):
        """Accuracy within threshold"""
        return (torch.abs(pred - true) < threshold).float().mean().item()

# ======================== Training Functions ========================

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Train one epoch"""
    model.train()
    running_losses = {
        'total_loss': 0, 'seg_loss': 0,
        'bleaching_loss': 0, 'coverage_loss': 0
    }

    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        targets = {
            'segmentation_mask': batch['segmentation_mask'].to(device),
            'bleaching_ratio': batch['bleaching_ratio'].to(device),
            'coral_coverage': batch['coral_coverage'].to(device)
        }

        optimizer.zero_grad()
        outputs = model(images)
        losses = criterion(outputs, targets)

        losses['total_loss'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Step scheduler after each batch for OneCycleLR
        if scheduler is not None:
            scheduler.step()

        for key in running_losses:
            running_losses[key] += losses[key].item()

        pbar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'seg': f"{losses['seg_loss'].item():.3f}",
            'bleach': f"{losses['bleaching_loss'].item():.3f}"
        })

    for key in running_losses:
        running_losses[key] /= len(dataloader)

    return running_losses

def validate_epoch(model, dataloader, criterion, device):
    """Validate one epoch"""
    model.eval()
    running_losses = {
        'total_loss': 0, 'seg_loss': 0,
        'bleaching_loss': 0, 'coverage_loss': 0
    }
    metrics = {
        'iou': [],
        'bleaching_mae': [], 'coverage_mae': [],
        'bleaching_acc': []
    }
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            images = batch['image'].to(device)
            targets = {
                'segmentation_mask': batch['segmentation_mask'].to(device),
                'bleaching_ratio': batch['bleaching_ratio'].to(device),
                'coral_coverage': batch['coral_coverage'].to(device)
            }

            outputs = model(images)
            losses = criterion(outputs, targets)

            for key in running_losses:
                running_losses[key] += losses[key].item()

            # Calculate metrics
            iou = MetricCalculator.calculate_iou(outputs['segmentation'], targets['segmentation_mask'])
            bleaching_mae = MetricCalculator.calculate_mae(outputs['bleaching_ratio'], targets['bleaching_ratio'])
            coverage_mae = MetricCalculator.calculate_mae(outputs['coral_coverage'], targets['coral_coverage'])
            bleaching_acc = MetricCalculator.calculate_accuracy_threshold(
                outputs['bleaching_ratio'],
                targets['bleaching_ratio'],
                threshold=0.1
            )

            metrics['iou'].append(iou)
            metrics['bleaching_mae'].append(bleaching_mae)
            metrics['coverage_mae'].append(coverage_mae)
            metrics['bleaching_acc'].append(bleaching_acc)
    
    for key in running_losses:
        running_losses[key] /= len(dataloader)
    
    for key in metrics:
        metrics[key] = np.mean(metrics[key])
    
    return running_losses, metrics

# ======================== Training Pipeline ========================

def train_improved_model(config):
    """Complete training pipeline for improved model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare dataset
    all_image_ids = [f.split('.')[0] for f in os.listdir(config['image_dir'])
                     if f.lower().endswith('.jpg')]
    
    # Split dataset with stratification
    train_ids, val_ids = train_test_split(
        all_image_ids, 
        test_size=0.15, 
        random_state=config['seed']
    )
    
    print(f"Dataset split: {len(train_ids)} train, {len(val_ids)} val")
    
    # Create datasets
    train_dataset = ImprovedCoralDataset(
        image_dir=config['image_dir'],
        bleached_mask_dir=config['bleached_mask_dir'],
        non_bleached_mask_dir=config['non_bleached_mask_dir'],
        image_ids=train_ids,
        transform=get_enhanced_transforms('train', config['img_size']),
        mode='train',
        filter_low_quality=config.get('filter_low_quality', True)
    )
    
    val_dataset = ImprovedCoralDataset(
        image_dir=config['image_dir'],
        bleached_mask_dir=config['bleached_mask_dir'],
        non_bleached_mask_dir=config['non_bleached_mask_dir'],
        image_ids=val_ids,
        transform=get_enhanced_transforms('val', config['img_size']),
        mode='val',
        filter_low_quality=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = ImprovedCoralModel(
        backbone_name=config['backbone'],
        num_classes=3,
        pretrained=True
    ).to(device)
    
    print(f"Model created: {config['backbone']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Loss function
    criterion = CombinedLoss(
        seg_weight=config.get('seg_weight', 2.0),
        dice_weight=config.get('dice_weight', 1.5),
        bleaching_weight=config.get('bleaching_weight', 2.0),
        coverage_weight=config.get('coverage_weight', 0.8)
    ).to(device)
    
    # Optimizer with differential learning rates
    backbone_params = []
    decoder_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'decoder' in name or 'aspp' in name or 'cbam' in name:
            decoder_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config['lr'] * 0.1},  # Lower LR for backbone
        {'params': decoder_params, 'lr': config['lr'] * 0.5},   # Medium LR for decoder
        {'params': head_params, 'lr': config['lr']}             # Full LR for heads
    ], weight_decay=config['weight_decay'])
    
    # Learning rate scheduler - more stable strategy
    # Use OneCycleLR for better training dynamics
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config['lr'] * 0.1, config['lr'] * 0.5, config['lr']],  # Match param groups
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 30% of training for warmup
        anneal_strategy='cos',
        div_factor=25.0,  # initial_lr = max_lr/25
        final_div_factor=1000.0  # min_lr = initial_lr/1000
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_iou = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'val_iou': [],
        'val_bleaching_mae': [], 'val_bleaching_acc': []
    }
    
    print(f"\nStarting training for {config['epochs']} epochs...\n")
    
    for epoch in range(config['epochs']):
        print(f"{'='*60}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        
        # Validate
        val_losses, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_losses['total_loss'])
        history['val_loss'].append(val_losses['total_loss'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_bleaching_mae'].append(val_metrics['bleaching_mae'])
        history['val_bleaching_acc'].append(val_metrics['bleaching_acc'])

        # Print summary with loss breakdown
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_losses['total_loss']:.4f} "
              f"[seg={train_losses['seg_loss']:.2f}, bl={train_losses['bleaching_loss']:.2f}]")
        print(f"  Val Loss:   {val_losses['total_loss']:.4f} "
              f"[seg={val_losses['seg_loss']:.2f}, bl={val_losses['bleaching_loss']:.2f}]")
        print(f"  Val IoU:    {val_metrics['iou']:.4f}")
        print(f"  Bleach MAE: {val_metrics['bleaching_mae']:.4f}")
        print(f"  Bleach Acc: {val_metrics['bleaching_acc']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save best model
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, 'best_improved_model.pth')
            print(f"Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Also save if best IoU
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_iou,
                'val_metrics': val_metrics,
                'config': config
            }, 'best_iou_model.pth')
            print(f"Saved best IoU model (IoU: {best_iou:.4f})")
    
    return model, history, val_ids

# ======================== Visualization ========================

def visualize_improved_predictions(model, dataset, device, num_samples=4):
    """Visualize predictions (without boundary detection)"""
    model.eval()

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))

    indices = random.sample(range(len(dataset)), num_samples)

    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]

            image = sample['image'].unsqueeze(0).to(device)
            outputs = model(image)

            seg_pred = outputs['segmentation'].argmax(dim=1).cpu().squeeze()
            bleaching_ratio_pred = outputs['bleaching_ratio'].cpu().item()

            # Denormalize image
            img_display = sample['image'].cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_display = img_display * std + mean
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)

            seg_true = sample['segmentation_mask'].cpu()
            bleaching_ratio_true = sample['bleaching_ratio'].item()

            # Plot
            axes[idx, 0].imshow(img_display)
            axes[idx, 0].set_title('Original Image')
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(seg_true, cmap='tab10', vmin=0, vmax=2)
            axes[idx, 1].set_title('True Segmentation')
            axes[idx, 1].axis('off')

            axes[idx, 2].imshow(seg_pred, cmap='tab10', vmin=0, vmax=2)
            axes[idx, 2].set_title('Pred Segmentation')
            axes[idx, 2].axis('off')

            # Bleaching visualization
            bleaching_vis_pred = np.zeros_like(seg_pred)
            bleaching_vis_pred[seg_pred == 2] = 1
            axes[idx, 3].imshow(bleaching_vis_pred, cmap='Reds', vmin=0, vmax=1)
            axes[idx, 3].set_title(f'Pred Bleaching\nTrue: {bleaching_ratio_true:.2%}\nPred: {bleaching_ratio_pred:.2%}')
            axes[idx, 3].axis('off')

    plt.tight_layout()
    plt.savefig('improved_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_improved_history(history):
    """Plot training history (without boundary F1)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # IoU
    axes[0, 1].plot(history['val_iou'], label='Val IoU', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('IoU', fontsize=12)
    axes[0, 1].set_title('Validation IoU', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Bleaching MAE
    axes[1, 0].plot(history['val_bleaching_mae'], label='Bleaching MAE', color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('MAE', fontsize=12)
    axes[1, 0].set_title('Bleaching Ratio MAE', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Bleaching Accuracy
    axes[1, 1].plot(history['val_bleaching_acc'], label='Bleaching Acc (±10%)', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy', fontsize=12)
    axes[1, 1].set_title('Bleaching Prediction Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()

    # Add summary text
    summary_text = f"""Training Summary (No Boundary Detection)

Best Val Loss: {min(history['val_loss']):.4f}
Best IoU: {max(history['val_iou']):.4f}
Best Bleach MAE: {min(history['val_bleaching_mae']):.4f}
Best Bleach Acc: {max(history['val_bleaching_acc']):.4f}

Final Epoch:
Val Loss: {history['val_loss'][-1]:.4f}
Val IoU: {history['val_iou'][-1]:.4f}"""

    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10,
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig('improved_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

# ======================== Main ========================

if __name__ == "__main__":
    # Enhanced configuration
    config = {
        # Data paths
        'image_dir': 'data/images',
        'bleached_mask_dir': 'data/masks_bleached',
        'non_bleached_mask_dir': 'data/masks_non_bleached',
        
        # Model
        'backbone': 'tf_efficientnet_b2_ns',  # EfficientNet-B2 with noisy student
        
        # Training (optimized for RTX 2060 - 6GB VRAM)
        'batch_size': 4,  # Optimal for 6GB VRAM with 512x512 images
        'epochs': 60,  # Reduced from 100 (training plateaus after epoch 40)
        'lr': 3e-4,  # Optimal learning rate
        'weight_decay': 1e-4,
        'img_size': 512,  # Full resolution for better results
        'num_workers': 4,  # Utilize CPU cores for data loading
        'seed': 42,

        # Loss weights (NO boundary weight - boundary detection removed)
        'seg_weight': 2.0,
        'dice_weight': 1.5,
        'bleaching_weight': 2.0,
        'coverage_weight': 0.8,
        
        # Data quality
        'filter_low_quality': True
    }
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
       
    # Train model
    model, history, val_ids = train_improved_model(config)
    
    # Plot results
    plot_improved_history(history)
    
    # Visualize predictions
    val_dataset = ImprovedCoralDataset(
        image_dir=config['image_dir'],
        bleached_mask_dir=config['bleached_mask_dir'],
        non_bleached_mask_dir=config['non_bleached_mask_dir'],
        image_ids=val_ids,
        transform=get_enhanced_transforms('val', config['img_size']),
        mode='val',
        filter_low_quality=False
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_improved_predictions(model, val_dataset, device, num_samples=6)
    
    print("\nTraining complete!")