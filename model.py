import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
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

# ======================== Dataset Definition ========================

class CoralBleachingDataset(Dataset):
    """Coral bleaching detection dataset"""
    
    def __init__(self, 
                 image_dir: str,
                 bleached_mask_dir: str,
                 non_bleached_mask_dir: str,
                 image_ids: List[str],
                 transform=None,
                 mode='train'):
        """
        Args:
            image_dir: Original image directory
            bleached_mask_dir: Bleached coral mask directory
            non_bleached_mask_dir: Non-bleached coral mask directory
            image_ids: List of image IDs
            transform: Data augmentation
            mode: 'train' or 'val'
        """
        self.image_dir = image_dir
        self.bleached_mask_dir = bleached_mask_dir
        self.non_bleached_mask_dir = non_bleached_mask_dir
        self.image_ids = image_ids
        self.transform = transform
        self.mode = mode
        
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
        
        # Convert to binary masks
        bleached_mask = (bleached_mask > 127).astype(np.float32)
        non_bleached_mask = (non_bleached_mask > 127).astype(np.float32)
        
        # Create 3-class segmentation mask: background(0), healthy coral(1), bleached coral(2)
        segmentation_mask = np.zeros_like(bleached_mask, dtype=np.float32)
        segmentation_mask[non_bleached_mask == 1] = 1
        segmentation_mask[bleached_mask == 1] = 2
        
        # Calculate bleaching metrics
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
            # Ensure mask is integer type
            if isinstance(segmentation_mask, torch.Tensor):
                segmentation_mask = segmentation_mask.long()
            else:
                segmentation_mask = segmentation_mask.astype(np.int64)
        
        # Convert to tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(segmentation_mask, torch.Tensor):
            segmentation_mask = torch.from_numpy(segmentation_mask.astype(np.int64)).long()
        
        return {
            'image': image,
            'segmentation_mask': segmentation_mask,
            'bleaching_ratio': torch.tensor(bleaching_ratio, dtype=torch.float32),
            'coral_coverage': torch.tensor(coral_coverage, dtype=torch.float32),
            'image_id': img_id
        }

# ======================== Model Definition ========================

class CoralBleachingBaseline(nn.Module):
    """Coral bleaching detection baseline model"""
    
    def __init__(self, 
                 backbone_name='efficientnet_b0',
                 num_classes=3,
                 pretrained=True):
        super().__init__()
        
        # Use timm library pretrained model as backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4]  # Get multi-scale features
        )
        
        # Get feature channels
        channels = self.backbone.feature_info.channels()
        
        # Segmentation decoder (simplified U-Net)
        self.decoder4 = self._make_decoder_block(channels[3], channels[2], 256)
        self.decoder3 = self._make_decoder_block(256 + channels[2], channels[1], 128)
        self.decoder2 = self._make_decoder_block(128 + channels[1], channels[0], 64)
        self.decoder1 = self._make_decoder_block(64 + channels[0], channels[0], 32)
        
        # Segmentation head
        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # Global feature pooling for regression tasks
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Bleaching ratio regression head
        self.bleaching_head = nn.Sequential(
            nn.Linear(channels[3], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output bleaching ratio between 0-1
        )
        
        # Coral coverage regression head
        self.coverage_head = nn.Sequential(
            nn.Linear(channels[3], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output coverage between 0-1
        )
    
    def _make_decoder_block(self, in_channels, skip_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        features = self.backbone(x)
        
        # Segmentation decoder
        d4 = self.decoder4(features[3])
        d4_up = F.interpolate(d4, size=features[2].shape[2:], mode='bilinear', align_corners=False)
        d3 = self.decoder3(torch.cat([d4_up, features[2]], dim=1))
        d3_up = F.interpolate(d3, size=features[1].shape[2:], mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d3_up, features[1]], dim=1))
        d2_up = F.interpolate(d2, size=features[0].shape[2:], mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d2_up, features[0]], dim=1))
        
        # Segmentation output
        seg_logits = self.seg_head(d1)
        seg_output = F.interpolate(seg_logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Global features for regression
        global_features = self.global_pool(features[3]).flatten(1)
        
        # Bleaching ratio and coverage prediction
        bleaching_ratio = self.bleaching_head(global_features)
        coral_coverage = self.coverage_head(global_features)
        
        return {
            'segmentation': seg_output,
            'bleaching_ratio': bleaching_ratio.squeeze(1),
            'coral_coverage': coral_coverage.squeeze(1)
        }

# ======================== Loss Function ========================

class CombinedLoss(nn.Module):
    """Combined loss function"""
    
    def __init__(self, 
                 seg_weight=1.0,
                 bleaching_weight=1.0,
                 coverage_weight=0.5):
        super().__init__()
        self.seg_weight = seg_weight
        self.bleaching_weight = bleaching_weight
        self.coverage_weight = coverage_weight
        
        # Use weighted cross entropy to handle class imbalance
        self.seg_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 1.0, 1.2]))
        self.regression_criterion = nn.MSELoss()
    
    def forward(self, outputs, targets):
        # Segmentation loss
        seg_loss = self.seg_criterion(
            outputs['segmentation'],
            targets['segmentation_mask']
        )
        
        # Bleaching ratio regression loss
        bleaching_loss = self.regression_criterion(
            outputs['bleaching_ratio'],
            targets['bleaching_ratio']
        )
        
        # Coverage regression loss
        coverage_loss = self.regression_criterion(
            outputs['coral_coverage'],
            targets['coral_coverage']
        )
        
        # Combined loss
        total_loss = (self.seg_weight * seg_loss + 
                     self.bleaching_weight * bleaching_loss + 
                     self.coverage_weight * coverage_loss)
        
        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'bleaching_loss': bleaching_loss,
            'coverage_loss': coverage_loss
        }

# ======================== Evaluation Metrics ========================

class MetricCalculator:
    """Calculate various evaluation metrics"""
    
    @staticmethod
    def calculate_iou(pred, target, num_classes=3):
        """Calculate IoU (Intersection over Union)"""
        ious = []
        pred = pred.argmax(dim=1)
        
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).float().sum()
            union = (pred_cls | target_cls).float().sum()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou.item())
        
        return np.mean(ious) if ious else 0.0
    
    @staticmethod
    def calculate_mae(pred, target):
        """Calculate Mean Absolute Error"""
        return torch.abs(pred - target).mean().item()
    
    @staticmethod
    def calculate_accuracy_threshold(pred, target, threshold=0.1):
        """Calculate accuracy within threshold"""
        correct = (torch.abs(pred - target) < threshold).float()
        return correct.mean().item()

# ======================== Training and Validation Functions ========================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    running_losses = {'total_loss': 0, 'seg_loss': 0, 'bleaching_loss': 0, 'coverage_loss': 0}
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Prepare data
        images = batch['image'].to(device)
        targets = {
            'segmentation_mask': batch['segmentation_mask'].to(device),
            'bleaching_ratio': batch['bleaching_ratio'].to(device),
            'coral_coverage': batch['coral_coverage'].to(device)
        }
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        losses = criterion(outputs, targets)
        
        # Backward pass
        losses['total_loss'].backward()
        optimizer.step()
        
        # Update statistics
        for key in running_losses:
            running_losses[key] += losses[key].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'bleaching': f"{losses['bleaching_loss'].item():.4f}"
        })
    
    # Calculate average losses
    for key in running_losses:
        running_losses[key] /= len(dataloader)
    
    return running_losses

def validate_epoch(model, dataloader, criterion, device):
    """Validate one epoch"""
    model.eval()
    running_losses = {'total_loss': 0, 'seg_loss': 0, 'bleaching_loss': 0, 'coverage_loss': 0}
    metrics = {'iou': [], 'bleaching_mae': [], 'coverage_mae': [], 'bleaching_acc': []}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            # Prepare data
            images = batch['image'].to(device)
            targets = {
                'segmentation_mask': batch['segmentation_mask'].to(device),
                'bleaching_ratio': batch['bleaching_ratio'].to(device),
                'coral_coverage': batch['coral_coverage'].to(device)
            }
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            losses = criterion(outputs, targets)
            
            # Update loss statistics
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
    
    # Calculate average values
    for key in running_losses:
        running_losses[key] /= len(dataloader)
    
    for key in metrics:
        metrics[key] = np.mean(metrics[key])
    
    return running_losses, metrics

# ======================== Data Augmentation ========================

def get_transforms(mode='train', img_size=512):
    """Get data augmentation pipeline"""
    
    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1),
                A.RandomGamma(gamma_limit=(80, 120), p=1),
            ], p=0.8),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1),
                A.MedianBlur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ], p=0.3),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

# ======================== Main Training Function ========================

def train_model(config):
    """Complete training pipeline"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare dataset
    # Assume all image filenames follow consistent format
    all_image_ids = [f.split('.')[0] for f in os.listdir(config['image_dir']) 
                     if f.endswith('.jpg')]
    
    # Split training and validation sets
    train_ids, val_ids = train_test_split(
        all_image_ids, 
        test_size=0.2, 
        random_state=config['seed']
    )
    
    print(f"Training samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")
    
    # Create datasets
    train_dataset = CoralBleachingDataset(
        config['image_dir'],
        config['bleached_mask_dir'],
        config['non_bleached_mask_dir'],
        train_ids,
        transform=get_transforms('train', config['img_size']),
        mode='train'
    )
    
    val_dataset = CoralBleachingDataset(
        config['image_dir'],
        config['bleached_mask_dir'],
        config['non_bleached_mask_dir'],
        val_ids,
        transform=get_transforms('val', config['img_size']),
        mode='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = CoralBleachingBaseline(
        backbone_name=config['backbone'],
        num_classes=3,
        pretrained=True
    ).to(device)
    
    # Loss function
    criterion = CombinedLoss(
        seg_weight=config['seg_weight'],
        bleaching_weight=config['bleaching_weight'],
        coverage_weight=config['coverage_weight']
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config['learning_rate'] * 0.01
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_bleaching_mae': [],
        'val_bleaching_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)
        
        # Training
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_losses, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_losses['total_loss'])
        history['val_loss'].append(val_losses['total_loss'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_bleaching_mae'].append(val_metrics['bleaching_mae'])
        history['val_bleaching_acc'].append(val_metrics['bleaching_acc'])
        
        # Print results
        print(f"Train Loss: {train_losses['total_loss']:.4f}")
        print(f"Val Loss: {val_losses['total_loss']:.4f}")
        print(f"Val IoU: {val_metrics['iou']:.4f}")
        print(f"Val Bleaching MAE: {val_metrics['bleaching_mae']:.4f}")
        print(f"Val Bleaching Acc (±10%): {val_metrics['bleaching_acc']:.4f}")
        
        # Save best model
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, 'best_coral_model.pth')
            print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
    
    # Load best model
    model.load_state_dict(best_model_state)

    return model, history, val_ids

# ======================== Visualization Functions ========================

def visualize_predictions(model, dataset, device, num_samples=4):
    """Visualize model prediction results"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    
    indices = random.sample(range(len(dataset)), num_samples)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]
            
            # Prepare input
            image = sample['image'].unsqueeze(0).to(device)
            
            # Prediction
            outputs = model(image)
            
            # Get prediction results
            seg_pred = outputs['segmentation'].argmax(dim=1).cpu().squeeze()
            bleaching_ratio_pred = outputs['bleaching_ratio'].cpu().item()
            coverage_pred = outputs['coral_coverage'].cpu().item()
            
            # Denormalize image for display
            img_display = sample['image'].cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_display = img_display * std + mean
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)
            
            # True labels
            seg_true = sample['segmentation_mask'].cpu()
            bleaching_ratio_true = sample['bleaching_ratio'].item()
            coverage_true = sample['coral_coverage'].item()
            
            # Plot
            axes[idx, 0].imshow(img_display)
            axes[idx, 0].set_title('Original Image')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(seg_true, cmap='viridis', vmin=0, vmax=2)
            axes[idx, 1].set_title('True Segmentation')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(seg_pred, cmap='viridis', vmin=0, vmax=2)
            axes[idx, 2].set_title('Predicted Segmentation')
            axes[idx, 2].axis('off')
            
            # Create bleaching visualization
            bleaching_vis_true = np.zeros_like(seg_true)
            bleaching_vis_true[seg_true == 2] = 1  # Bleached areas
            axes[idx, 3].imshow(bleaching_vis_true, cmap='Reds', vmin=0, vmax=1)
            axes[idx, 3].set_title(f'True Bleaching\n(Ratio: {bleaching_ratio_true:.2%})')
            axes[idx, 3].axis('off')
            
            bleaching_vis_pred = np.zeros_like(seg_pred)
            bleaching_vis_pred[seg_pred == 2] = 1  # Predicted bleached areas
            axes[idx, 4].imshow(bleaching_vis_pred, cmap='Reds', vmin=0, vmax=1)
            axes[idx, 4].set_title(f'Pred Bleaching\n(Ratio: {bleaching_ratio_pred:.2%})')
            axes[idx, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150)
    plt.show()

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU curve
    axes[0, 1].plot(history['val_iou'], label='Val IoU', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Validation IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Bleaching MAE curve
    axes[1, 0].plot(history['val_bleaching_mae'], label='Bleaching MAE', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('Bleaching Ratio MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Bleaching accuracy curve
    axes[1, 1].plot(history['val_bleaching_acc'], label='Bleaching Acc (±10%)', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Bleaching Prediction Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()

# ======================== Main Function ========================

if __name__ == "__main__":
    # Configuration parameters
    config = {
        # Data paths
        'image_dir': 'data/images',
        'bleached_mask_dir': 'data/masks_bleached',
        'non_bleached_mask_dir': 'data/masks_non_bleached',
        
        # Model parameters
        'backbone': 'efficientnet_b0',  # Options: efficientnet_b0-b7, resnet50, etc.
        'img_size': 512,
        
        # Training parameters
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        
        # Loss weights
        'seg_weight': 1.0,
        'bleaching_weight': 1.0,
        'coverage_weight': 0.5,
        
        # Others
        'num_workers': 4,
        'seed': 42
    }
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    
    # Train model
    model, history, val_ids = train_model(config)
    
    # Plot training history
    plot_training_history(history)
    
    # Visualize prediction results
    val_dataset = CoralBleachingDataset(
        config['image_dir'],
        config['bleached_mask_dir'], 
        config['non_bleached_mask_dir'],
        val_ids,  # Need to get from train_model function
        transform=get_transforms('val', config['img_size'])
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_predictions(model, val_dataset, device)
    
    print("\nTraining completed successfully!")