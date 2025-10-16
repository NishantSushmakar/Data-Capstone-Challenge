"""
Multi-Architecture Comparison for Coral Bleaching Detection
Trains 3 CNN architectures and compares performance & carbon emissions
"""

import torch
import json
import os
import pandas as pd
from improved_model import *
from carbon_tracking_integration import (
    CarbonAwareTrainer,
    ModelEfficiencyAnalyzer
)

ARCHITECTURES = {
    'resnet34': {
        'name': 'resnet34',
        'description': 'ResNet-34 - Classic CNN baseline'
    },
    'efficientnet_b0': {
        'name': 'tf_efficientnet_b0_ns',
        'description': 'EfficientNet-B0 - Lightweight baseline'
    },
    'efficientnet_b2': {
        'name': 'tf_efficientnet_b2_ns',
        'description': 'EfficientNet-B2 - Improved model with ASPP & CBAM'
    }
}

def train_single_architecture(arch_key, base_config, device):
    """Train a single architecture from scratch with carbon tracking"""

    arch_info = ARCHITECTURES[arch_key]
    backbone_name = arch_info['name']

    print("\n" + "="*80)
    print(f"TRAINING: {arch_info['description']}")
    print(f"Backbone: {backbone_name}")
    print("="*80 + "\n")

    config = base_config.copy()
    config['backbone'] = backbone_name

    # Initialize carbon tracker
    carbon_tracker = CarbonAwareTrainer(
        project_name=f"coral_bleaching_{arch_key}",
        country_iso_code="NLD",
        region="North Holland",
        tracking_mode="offline",
        output_dir="./carbon_reports"
    )

    # Prepare dataset
    all_image_ids = [f.split('.')[0] for f in os.listdir(config['image_dir'])
                     if f.lower().endswith('.jpg')]

    train_ids, val_ids = train_test_split(
        all_image_ids,
        test_size=0.15,
        random_state=config['seed']
    )

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

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True
    )

    # Create model
    model = ImprovedCoralModel(
        backbone_name=backbone_name, num_classes=3, pretrained=True
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Analyze model efficiency
    print("\nAnalyzing model efficiency...")
    efficiency_metrics = ModelEfficiencyAnalyzer.comprehensive_analysis(
        model=model,
        input_size=(1, 3, config['img_size'], config['img_size']),
        device=device
    )

    # Loss and optimizer
    criterion = CombinedLoss(
        seg_weight=config.get('seg_weight', 2.0),
        dice_weight=config.get('dice_weight', 1.5),
        bleaching_weight=config.get('bleaching_weight', 2.0),
        coverage_weight=config.get('coverage_weight', 0.8)
    ).to(device)

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
        {'params': backbone_params, 'lr': config['lr'] * 0.1},
        {'params': decoder_params, 'lr': config['lr'] * 0.5},
        {'params': head_params, 'lr': config['lr']}
    ], weight_decay=config['weight_decay'])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config['lr'] * 0.1, config['lr'] * 0.5, config['lr']],
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )

    # Start carbon tracking
    carbon_tracker.start_training()

    # Training loop
    best_val_loss = float('inf')
    best_iou = 0.0
    epochs_without_improvement = 0
    patience = 10

    history = {
        'train_loss': [], 'val_loss': [],
        'val_iou': [], 'val_bleaching_mae': [], 'val_bleaching_acc': []
    }

    print(f"\nStarting training (max {config['epochs']} epochs, early stop patience={patience})...\n")

    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}/{config['epochs']}")

        train_losses = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        val_losses, val_metrics = validate_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_losses['total_loss'])
        history['val_loss'].append(val_losses['total_loss'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_bleaching_mae'].append(val_metrics['bleaching_mae'])
        history['val_bleaching_acc'].append(val_metrics['bleaching_acc'])

        print(f"  Train Loss: {train_losses['total_loss']:.4f}, Val Loss: {val_losses['total_loss']:.4f}")
        print(f"  Val IoU: {val_metrics['iou']:.4f}, Bleach Acc: {val_metrics['bleaching_acc']:.4f}\n")

        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            epochs_without_improvement = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, f'best_model_{arch_key}.pth')
        else:
            epochs_without_improvement += 1

        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best val loss: {best_val_loss:.4f}, Best IoU: {best_iou:.4f}\n")
            break

    training_emissions = carbon_tracker.stop_training()

    # Measure inference emissions
    print("\nMeasuring inference carbon footprint...")
    inference_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        num_workers=4, pin_memory=True
    )

    inference_emissions = carbon_tracker.track_inference(
        model=model,
        dataloader=inference_loader,
        device=device,
        num_samples=min(1000, len(val_dataset))
    )

    carbon_report = carbon_tracker.save_detailed_report(
        filename=f"carbon_report_{arch_key}.json"
    )

    # Compile results
    results = {
        'architecture': arch_key,
        'backbone_name': backbone_name,
        'description': arch_info['description'],
        'parameters_millions': efficiency_metrics['params_millions'],
        'flops_billions': efficiency_metrics['flops_billions'],
        'model_size_mb': efficiency_metrics['model_size_mb'],
        'best_val_loss': best_val_loss,
        'best_val_iou': best_iou,
        'final_bleaching_mae': history['val_bleaching_mae'][-1],
        'final_bleaching_acc': history['val_bleaching_acc'][-1],
        'training_emissions_kg_co2': training_emissions,
        'training_duration_hours': carbon_report['training']['duration_hours'],
        'inference_emissions_g_co2_per_image': carbon_report['inference']['emissions_per_prediction_g'],
        'inference_latency_ms': efficiency_metrics.get('avg_latency_ms', efficiency_metrics.get('latency_ms', 0)),
        'history': history,
        'efficiency_metrics': efficiency_metrics,
        'carbon_report': carbon_report
    }

    # Cleanup
    if os.path.exists(f'best_model_{arch_key}.pth'):
        os.remove(f'best_model_{arch_key}.pth')

    print(f"\n✓ Completed training for {arch_key}")

    return results


def compare_all_architectures(base_config):
    """Train all architectures and compare"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    all_results = []

    for arch_key in ARCHITECTURES.keys():
        try:
            results = train_single_architecture(arch_key, base_config, device)
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"\nERROR training {arch_key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("ERROR: No models trained successfully!")
        return None, None

    # Create comparison table
    comparison_df = pd.DataFrame([{
        'Architecture': r['architecture'],
        'Description': r['description'],
        'Val IoU': f"{r['best_val_iou']:.4f}",
        'Bleach Acc': f"{r['final_bleaching_acc']:.4f}",
        'Bleach MAE': f"{r['final_bleaching_mae']:.4f}",
        'Params (M)': f"{r['parameters_millions']:.2f}",
        'FLOPs (G)': f"{r['flops_billions']:.2f}",
        'Size (MB)': f"{r['model_size_mb']:.2f}",
        'Train CO2 (kg)': f"{r['training_emissions_kg_co2']:.4f}",
        'Train Time (h)': f"{r['training_duration_hours']:.2f}",
        'Infer CO2 (g)': f"{r['inference_emissions_g_co2_per_image']:.6f}",
        'Latency (ms)': f"{r['inference_latency_ms']:.2f}"
    } for r in all_results])

    comparison_df.to_csv('architecture_comparison.csv', index=False)

    print("\n" + "="*80)
    print("ARCHITECTURE COMPARISON: PERFORMANCE & CARBON EMISSIONS")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("\n✓ Saved to: architecture_comparison.csv")

    with open('detailed_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("✓ Saved to: detailed_comparison_results.json")

    # Detailed comparison
    print("\n" + "="*80)
    print("DETAILED COMPARISON - PERFORMANCE & CARBON EMISSIONS")
    print("="*80)

    best_perf = max(all_results, key=lambda x: x['best_val_iou'])
    best_carbon = min(all_results, key=lambda x: x['training_emissions_kg_co2'])

    print(f"\nBEST PERFORMANCE: {best_perf['architecture']}")
    print(f"  IoU: {best_perf['best_val_iou']:.4f}")
    print(f"  Training CO2: {best_perf['training_emissions_kg_co2']:.4f} kg")

    print(f"\nLOWEST TRAINING CARBON: {best_carbon['architecture']}")
    print(f"  IoU: {best_carbon['best_val_iou']:.4f}")
    print(f"  Training CO2: {best_carbon['training_emissions_kg_co2']:.4f} kg")

    # Carbon efficiency
    print(f"\nCARBON EFFICIENCY (IoU per kg CO2):")
    for r in all_results:
        eff = r['best_val_iou'] / r['training_emissions_kg_co2']
        print(f"  {r['architecture']}: {eff:.2f} IoU/kg")

    print("\n" + "="*80)

    # Cleanup individual reports
    print("\nCleaning up intermediate files...")
    for arch_key in ARCHITECTURES.keys():
        report_file = f"carbon_report_{arch_key}.json"
        if os.path.exists(report_file):
            os.remove(report_file)

    print("\n✓ Comparison complete!")

    return all_results, comparison_df


if __name__ == "__main__":
    base_config = {
        'image_dir': 'data/images',
        'bleached_mask_dir': 'data/masks_bleached',
        'non_bleached_mask_dir': 'data/masks_non_bleached',
        'batch_size': 4,
        'epochs': 50,
        'lr': 3e-4,
        'weight_decay': 5e-4,
        'img_size': 512,
        'num_workers': 4,
        'seed': 42,
        'seg_weight': 2.0,
        'dice_weight': 1.5,
        'bleaching_weight': 3.0,
        'coverage_weight': 0.8,
        'filter_low_quality': True
    }

    torch.manual_seed(base_config['seed'])
    np.random.seed(base_config['seed'])
    random.seed(base_config['seed'])

    print("="*80)
    print("CNN ARCHITECTURE COMPARISON: PERFORMANCE & CARBON EMISSIONS")
    print("="*80)
    print("\nWill train 3 architectures from scratch:")
    for key, info in ARCHITECTURES.items():
        print(f"  - {key}: {info['description']}")
    print("\nEach model trained with same config, carbon tracking enabled")
    print("Expected runtime: 2-6 hours")
    print("="*80 + "\n")

    all_results, comparison_df = compare_all_architectures(base_config)
