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

# ======================== 数据集定义 ========================

class CoralBleachingDataset(Dataset):
    """珊瑚白化检测数据集"""
    
    def __init__(self, 
                 image_dir: str,
                 bleached_mask_dir: str,
                 non_bleached_mask_dir: str,
                 image_ids: List[str],
                 transform=None,
                 mode='train'):
        """
        Args:
            image_dir: 原始图像目录
            bleached_mask_dir: 白化珊瑚掩码目录
            non_bleached_mask_dir: 非白化珊瑚掩码目录
            image_ids: 图像ID列表
            transform: 数据增强
            mode: 'train' 或 'val'
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
        
        # 加载图像和掩码
        image_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        bleached_mask_path = os.path.join(self.bleached_mask_dir, f"{img_id}_bleached.png")
        non_bleached_mask_path = os.path.join(self.non_bleached_mask_dir, f"{img_id}_non_bleached.png")
        
        image = np.array(Image.open(image_path).convert('RGB'))
        bleached_mask = np.array(Image.open(bleached_mask_path).convert('L'))
        non_bleached_mask = np.array(Image.open(non_bleached_mask_path).convert('L'))
        
        # 转换为二值掩码
        bleached_mask = (bleached_mask > 127).astype(np.float32)
        non_bleached_mask = (non_bleached_mask > 127).astype(np.float32)
        
        # 创建3类分割掩码：背景(0)、健康珊瑚(1)、白化珊瑚(2)
        segmentation_mask = np.zeros_like(bleached_mask, dtype=np.float32)
        segmentation_mask[non_bleached_mask == 1] = 1
        segmentation_mask[bleached_mask == 1] = 2
        
        # 计算白化指标
        total_coral_pixels = (bleached_mask == 1).sum() + (non_bleached_mask == 1).sum()
        if total_coral_pixels > 0:
            bleaching_ratio = (bleached_mask == 1).sum() / total_coral_pixels
            coral_coverage = total_coral_pixels / (bleached_mask.shape[0] * bleached_mask.shape[1])
        else:
            bleaching_ratio = 0.0
            coral_coverage = 0.0
        
        # 数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=segmentation_mask)
            image = transformed['image']
            segmentation_mask = transformed['mask']
            # 确保mask是整数类型
            if isinstance(segmentation_mask, torch.Tensor):
                segmentation_mask = segmentation_mask.long()
            else:
                segmentation_mask = segmentation_mask.astype(np.int64)
        
        # 转换为tensor
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

# ======================== 模型定义 ========================

class CoralBleachingBaseline(nn.Module):
    """珊瑚白化检测基线模型"""
    
    def __init__(self, 
                 backbone_name='efficientnet_b0',
                 num_classes=3,
                 pretrained=True):
        super().__init__()
        
        # 使用timm库的预训练模型作为backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4]  # 获取多尺度特征
        )
        
        # 获取特征通道数
        channels = self.backbone.feature_info.channels()
        
        # 分割解码器（简化版U-Net）
        self.decoder4 = self._make_decoder_block(channels[3], channels[2], 256)
        self.decoder3 = self._make_decoder_block(256 + channels[2], channels[1], 128)
        self.decoder2 = self._make_decoder_block(128 + channels[1], channels[0], 64)
        self.decoder1 = self._make_decoder_block(64 + channels[0], channels[0], 32)
        
        # 分割头
        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # 全局特征池化用于回归任务
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 白化比例回归头
        self.bleaching_head = nn.Sequential(
            nn.Linear(channels[3], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出0-1之间的白化比例
        )
        
        # 珊瑚覆盖率回归头
        self.coverage_head = nn.Sequential(
            nn.Linear(channels[3], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出0-1之间的覆盖率
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
        # 编码器
        features = self.backbone(x)
        
        # 分割解码器
        d4 = self.decoder4(features[3])
        d4_up = F.interpolate(d4, size=features[2].shape[2:], mode='bilinear', align_corners=False)
        d3 = self.decoder3(torch.cat([d4_up, features[2]], dim=1))
        d3_up = F.interpolate(d3, size=features[1].shape[2:], mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d3_up, features[1]], dim=1))
        d2_up = F.interpolate(d2, size=features[0].shape[2:], mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d2_up, features[0]], dim=1))
        
        # 分割输出
        seg_logits = self.seg_head(d1)
        seg_output = F.interpolate(seg_logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 全局特征用于回归
        global_features = self.global_pool(features[3]).flatten(1)
        
        # 白化比例和覆盖率预测
        bleaching_ratio = self.bleaching_head(global_features)
        coral_coverage = self.coverage_head(global_features)
        
        return {
            'segmentation': seg_output,
            'bleaching_ratio': bleaching_ratio.squeeze(1),
            'coral_coverage': coral_coverage.squeeze(1)
        }

# ======================== 损失函数 ========================

class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(self, 
                 seg_weight=1.0,
                 bleaching_weight=1.0,
                 coverage_weight=0.5):
        super().__init__()
        self.seg_weight = seg_weight
        self.bleaching_weight = bleaching_weight
        self.coverage_weight = coverage_weight
        
        # 使用加权交叉熵处理类别不平衡
        self.seg_criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 1.0, 1.2]))
        self.regression_criterion = nn.MSELoss()
    
    def forward(self, outputs, targets):
        # 分割损失
        seg_loss = self.seg_criterion(
            outputs['segmentation'],
            targets['segmentation_mask']
        )
        
        # 白化比例回归损失
        bleaching_loss = self.regression_criterion(
            outputs['bleaching_ratio'],
            targets['bleaching_ratio']
        )
        
        # 覆盖率回归损失
        coverage_loss = self.regression_criterion(
            outputs['coral_coverage'],
            targets['coral_coverage']
        )
        
        # 组合损失
        total_loss = (self.seg_weight * seg_loss + 
                     self.bleaching_weight * bleaching_loss + 
                     self.coverage_weight * coverage_loss)
        
        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'bleaching_loss': bleaching_loss,
            'coverage_loss': coverage_loss
        }

# ======================== 评估指标 ========================

class MetricCalculator:
    """计算各种评估指标"""
    
    @staticmethod
    def calculate_iou(pred, target, num_classes=3):
        """计算IoU (Intersection over Union)"""
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
        """计算平均绝对误差"""
        return torch.abs(pred - target).mean().item()
    
    @staticmethod
    def calculate_accuracy_threshold(pred, target, threshold=0.1):
        """计算阈值内的准确率"""
        correct = (torch.abs(pred - target) < threshold).float()
        return correct.mean().item()

# ======================== 训练和验证函数 ========================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_losses = {'total_loss': 0, 'seg_loss': 0, 'bleaching_loss': 0, 'coverage_loss': 0}
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # 准备数据
        images = batch['image'].to(device)
        targets = {
            'segmentation_mask': batch['segmentation_mask'].to(device),
            'bleaching_ratio': batch['bleaching_ratio'].to(device),
            'coral_coverage': batch['coral_coverage'].to(device)
        }
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        
        # 计算损失
        losses = criterion(outputs, targets)
        
        # 反向传播
        losses['total_loss'].backward()
        optimizer.step()
        
        # 更新统计
        for key in running_losses:
            running_losses[key] += losses[key].item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'bleaching': f"{losses['bleaching_loss'].item():.4f}"
        })
    
    # 计算平均损失
    for key in running_losses:
        running_losses[key] /= len(dataloader)
    
    return running_losses

def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    running_losses = {'total_loss': 0, 'seg_loss': 0, 'bleaching_loss': 0, 'coverage_loss': 0}
    metrics = {'iou': [], 'bleaching_mae': [], 'coverage_mae': [], 'bleaching_acc': []}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            # 准备数据
            images = batch['image'].to(device)
            targets = {
                'segmentation_mask': batch['segmentation_mask'].to(device),
                'bleaching_ratio': batch['bleaching_ratio'].to(device),
                'coral_coverage': batch['coral_coverage'].to(device)
            }
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            losses = criterion(outputs, targets)
            
            # 更新损失统计
            for key in running_losses:
                running_losses[key] += losses[key].item()
            
            # 计算指标
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
    
    # 计算平均值
    for key in running_losses:
        running_losses[key] /= len(dataloader)
    
    for key in metrics:
        metrics[key] = np.mean(metrics[key])
    
    return running_losses, metrics

# ======================== 数据增强 ========================

def get_transforms(mode='train', img_size=512):
    """获取数据增强pipeline"""
    
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

# ======================== 主训练函数 ========================

def train_model(config):
    """完整的训练流程"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 准备数据集
    # 假设所有图像文件名格式一致
    all_image_ids = [f.split('.')[0] for f in os.listdir(config['image_dir']) 
                     if f.endswith('.jpg')]
    
    # 划分训练集和验证集
    train_ids, val_ids = train_test_split(
        all_image_ids, 
        test_size=0.2, 
        random_state=config['seed']
    )
    
    print(f"Training samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")
    
    # 创建数据集
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
    
    # 创建数据加载器
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
    
    # 创建模型
    model = CoralBleachingBaseline(
        backbone_name=config['backbone'],
        num_classes=3,
        pretrained=True
    ).to(device)
    
    # 损失函数
    criterion = CombinedLoss(
        seg_weight=config['seg_weight'],
        bleaching_weight=config['bleaching_weight'],
        coverage_weight=config['coverage_weight']
    ).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config['learning_rate'] * 0.01
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_bleaching_mae': [],
        'val_bleaching_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # 训练循环
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)
        
        # 训练
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_losses, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_losses['total_loss'])
        history['val_loss'].append(val_losses['total_loss'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_bleaching_mae'].append(val_metrics['bleaching_mae'])
        history['val_bleaching_acc'].append(val_metrics['bleaching_acc'])
        
        # 打印结果
        print(f"Train Loss: {train_losses['total_loss']:.4f}")
        print(f"Val Loss: {val_losses['total_loss']:.4f}")
        print(f"Val IoU: {val_metrics['iou']:.4f}")
        print(f"Val Bleaching MAE: {val_metrics['bleaching_mae']:.4f}")
        print(f"Val Bleaching Acc (±10%): {val_metrics['bleaching_acc']:.4f}")
        
        # 保存最佳模型
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
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)

    return model, history, val_ids

# ======================== 可视化函数 ========================

def visualize_predictions(model, dataset, device, num_samples=4):
    """可视化模型预测结果"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    
    indices = random.sample(range(len(dataset)), num_samples)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]
            
            # 准备输入
            image = sample['image'].unsqueeze(0).to(device)
            
            # 预测
            outputs = model(image)
            
            # 获取预测结果
            seg_pred = outputs['segmentation'].argmax(dim=1).cpu().squeeze()
            bleaching_ratio_pred = outputs['bleaching_ratio'].cpu().item()
            coverage_pred = outputs['coral_coverage'].cpu().item()
            
            # 反归一化图像用于显示
            img_display = sample['image'].cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_display = img_display * std + mean
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)
            
            # 真实标签
            seg_true = sample['segmentation_mask'].cpu()
            bleaching_ratio_true = sample['bleaching_ratio'].item()
            coverage_true = sample['coral_coverage'].item()
            
            # 绘制
            axes[idx, 0].imshow(img_display)
            axes[idx, 0].set_title('Original Image')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(seg_true, cmap='viridis', vmin=0, vmax=2)
            axes[idx, 1].set_title('True Segmentation')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(seg_pred, cmap='viridis', vmin=0, vmax=2)
            axes[idx, 2].set_title('Predicted Segmentation')
            axes[idx, 2].axis('off')
            
            # 创建白化可视化
            bleaching_vis_true = np.zeros_like(seg_true)
            bleaching_vis_true[seg_true == 2] = 1  # 白化区域
            axes[idx, 3].imshow(bleaching_vis_true, cmap='Reds', vmin=0, vmax=1)
            axes[idx, 3].set_title(f'True Bleaching\n(Ratio: {bleaching_ratio_true:.2%})')
            axes[idx, 3].axis('off')
            
            bleaching_vis_pred = np.zeros_like(seg_pred)
            bleaching_vis_pred[seg_pred == 2] = 1  # 预测的白化区域
            axes[idx, 4].imshow(bleaching_vis_pred, cmap='Reds', vmin=0, vmax=1)
            axes[idx, 4].set_title(f'Pred Bleaching\n(Ratio: {bleaching_ratio_pred:.2%})')
            axes[idx, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150)
    plt.show()

def plot_training_history(history):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU曲线
    axes[0, 1].plot(history['val_iou'], label='Val IoU', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Validation IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 白化MAE曲线
    axes[1, 0].plot(history['val_bleaching_mae'], label='Bleaching MAE', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('Bleaching Ratio MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 白化准确率曲线
    axes[1, 1].plot(history['val_bleaching_acc'], label='Bleaching Acc (±10%)', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Bleaching Prediction Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()

# ======================== 主函数 ========================

if __name__ == "__main__":
    # 配置参数
    config = {
        # 数据路径
        'image_dir': 'data/images',
        'bleached_mask_dir': 'data/masks_bleached',
        'non_bleached_mask_dir': 'data/masks_non_bleached',
        
        # 模型参数
        'backbone': 'efficientnet_b0',  # 可选: efficientnet_b0-b7, resnet50等
        'img_size': 512,
        
        # 训练参数
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        
        # 损失权重
        'seg_weight': 1.0,
        'bleaching_weight': 1.0,
        'coverage_weight': 0.5,
        
        # 其他
        'num_workers': 4,
        'seed': 42
    }
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    
    # 训练模型
    model, history, val_ids = train_model(config)
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 可视化预测结果
    val_dataset = CoralBleachingDataset(
        config['image_dir'],
        config['bleached_mask_dir'], 
        config['non_bleached_mask_dir'],
        val_ids,  # 需要从train_model函数中获取
        transform=get_transforms('val', config['img_size'])
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_predictions(model, val_dataset, device)
    
    print("\nTraining completed successfully!")