"""
Modified Training Script with Carbon Tracking
Integrates environmental cost monitoring into the existing coral bleaching model

1. Real-time carbon emission tracking during training
2. Model efficiency metrics (FLOPs, params, latency)
3. Inference cost measurement
4. Comprehensive SLE analysis report
"""

import torch
import sys
import json
from improved_model import *  # Import all your existing code
from carbon_tracking_integration import (
    CarbonAwareTrainer,
    ModelEfficiencyAnalyzer,
    EnvironmentalImpactAnalyzer
)

def train_with_carbon_tracking(config):
    """Enhanced training pipeline with carbon tracking"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==================== Initialize Carbon Tracking ====================
    carbon_tracker = CarbonAwareTrainer(
        project_name=f"coral_bleaching_{config['backbone']}",
        country_iso_code="NLD",  # Netherlands - adjust for your location
        region="North Holland",
        tracking_mode="offline",  # Use "online" if you want real-time grid data
        output_dir="./carbon_reports"
    )
    
    # ==================== Prepare Dataset (same as before) ====================
    all_image_ids = [f.split('.')[0] for f in os.listdir(config['image_dir']) 
                     if f.endswith('.jpg')]
    
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
    
    # ==================== Create Model ====================
    model = ImprovedCoralModel(
        backbone_name=config['backbone'],
        num_classes=3,
        pretrained=True
    ).to(device)
    
    print(f"Model created: {config['backbone']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # ==================== Model Efficiency Analysis ====================
    print("\n[SEARCH] Analyzing model efficiency before training...")
    efficiency_metrics = ModelEfficiencyAnalyzer.comprehensive_analysis(
        model=model,
        input_size=(1, 3, config['img_size'], config['img_size']),
        device=device
    )
    
    # Loss function
    criterion = CombinedLoss(
        seg_weight=config.get('seg_weight', 2.0),
        dice_weight=config.get('dice_weight', 1.5),
        bleaching_weight=config.get('bleaching_weight', 2.0),
        coverage_weight=config.get('coverage_weight', 0.8)
    ).to(device)
    
    # Optimizer
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
    
    # Scheduler
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
    
    # ==================== START CARBON TRACKING ====================
    carbon_tracker.start_training()
    
    # ==================== Training Loop ====================
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
        train_losses = train_epoch(model, train_loader, criterion, optimizer, 
                                  device, scheduler)
        
        # Validate
        val_losses, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_losses['total_loss'])
        history['val_loss'].append(val_losses['total_loss'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_bleaching_mae'].append(val_metrics['bleaching_mae'])
        history['val_bleaching_acc'].append(val_metrics['bleaching_acc'])

        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_losses['total_loss']:.4f}")
        print(f"  Val Loss:   {val_losses['total_loss']:.4f}")
        print(f"  Val IoU:    {val_metrics['iou']:.4f}")
        print(f"  Bleach MAE: {val_metrics['bleaching_mae']:.4f}")
        print(f"  Bleach Acc: {val_metrics['bleaching_acc']:.4f}\n")
        
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
            print(f"[OK] Saved best model (val_loss: {best_val_loss:.4f})")
        
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_iou': best_iou,
                'val_metrics': val_metrics,
                'config': config
            }, 'best_iou_model.pth')
            print(f"[OK] Saved best IoU model (IoU: {best_iou:.4f})")
    
    # ==================== STOP CARBON TRACKING ====================
    training_emissions = carbon_tracker.stop_training()
    
    # ==================== Inference Carbon Tracking ====================
    print("\n[SEARCH] Measuring inference carbon footprint...")
    inference_emissions = carbon_tracker.track_inference(
        model=model,
        dataloader=val_loader,
        device=device,
        num_samples=1000  # Test on 1000 images
    )
    
    # Save detailed carbon report
    carbon_report = carbon_tracker.save_detailed_report(
        filename="coral_bleaching_carbon_report.json"
    )
    
    # ==================== SLE Analysis ====================
    print("\n[CHART] Generating SLE environmental impact analysis...")
    
    impact_analyzer = EnvironmentalImpactAnalyzer()
    
    # Define assumptions about traditional methods for your SLE essay
    traditional_assumptions = {
        'emissions_kg_per_survey': 50.0,  # Boat fuel + equipment
        'images_per_survey': 1000,         # Images per survey
        'survey_frequency_per_year': 12    # Monthly monitoring
    }
    
    # Generate comprehensive SLE report
    sle_report = impact_analyzer.generate_sle_report(
        carbon_report=carbon_report,
        efficiency_metrics=efficiency_metrics,
        traditional_method_assumptions=traditional_assumptions
    )
    
    # Save SLE report
    with open('sle_environmental_impact_report.json', 'w') as f:
        json.dump(sle_report, f, indent=2)
    
    print("\n[OK] SLE report saved to: sle_environmental_impact_report.json")
    
    # ==================== Print Key SLE Findings ====================
    print("\n" + "="*70)
    print("[EARTH] KEY FINDINGS FOR YOUR SLE ESSAY")
    print("="*70)
    
    print("\n QUESTION 1: Computational Footprint")
    print("-" * 70)
    q1 = sle_report['question_1_computational_footprint']
    print(f"Training Emissions: {q1['training_phase']['total_emissions_kg_co2eq']:.4f} kg CO2eq")
    print(f"Training Duration: {q1['training_phase']['duration_hours']:.2f} hours")
    print(f"Model Parameters: {q1['training_phase']['model_complexity']['parameters_millions']:.2f}M")
    print(f"FLOPs: {q1['training_phase']['model_complexity']['flops_billions']:.2f} GFLOPs")
    
    equiv = q1['training_phase']['equivalent_comparisons']
    print(f"\nEquivalent to:")
    print(f"   Driving {equiv['equivalent_km_driven']:.1f} km in a car")
    print(f"   {equiv['trees_year_to_offset']:.2f} trees needed for 1 year to offset")
    print(f"   {equiv['smartphone_charges']:.0f} smartphone charges")
    
    print(f"\nPer-Prediction Cost: {q1['deployment_phase']['emissions_per_prediction_g']:.6f} g CO2")
    print(f"Inference Latency: {q1['deployment_phase']['latency_ms']:.2f} ms")
    
    print("\n QUESTION 3: Cost-Benefit Break-Even Analysis")
    print("-" * 70)
    q3 = sle_report['question_3_cost_benefit_analysis']
    print(q3['verdict'])
    if q3['break_even_images'] != float('inf'):
        print(f"\nBreak-even point: {q3['break_even_images']:.0f} images")
        print(f"  = {q3['break_even_surveys']:.1f} traditional surveys")
        print(f"\nCost comparison:")
        print(f"   AI training (one-time): {q3['ai_training_cost_kg']:.4f} kg CO2")
        print(f"   AI per image: {q3['ai_emissions_per_image_kg']*1000:.6f} g CO2")
        print(f"   Traditional per image: {q3['traditional_emissions_per_image_kg']*1000:.6f} g CO2")
        print(f"   Savings per image: {q3['savings_per_image_kg']*1000:.6f} g CO2")
    
    print("\n QUESTION 4: Optimization Recommendations")
    print("-" * 70)
    for i, rec in enumerate(sle_report['question_4_optimization_recommendations'], 1):
        print(f"\n{i}. {rec['category']}")
        print(f"   Issue: {rec['issue']}")
        print(f"   Suggestion: {rec['suggestion']}")
        print(f"   Potential reduction: {rec['potential_reduction']}")
    
    print("\n" + "="*70)
    
    # ==================== Visualizations ====================
    print("\n[CHART] Creating visualizations for SLE essay...")
    
    # Original training history
    plot_improved_history(history)
    
    # SLE-specific visualizations
    impact_analyzer.visualize_analysis(
        carbon_report=carbon_report,
        break_even_analysis=q3,
        save_path='sle_environmental_analysis.png'
    )
    
    # Model predictions
    visualize_improved_predictions(model, val_dataset, device, num_samples=6)
    
    print("\n[OK] All analyses complete! Files generated:")
    print("    coral_bleaching_carbon_report.json")
    print("    sle_environmental_impact_report.json")
    print("    sle_environmental_analysis.png")
    print("    improved_training_history.png")
    print("    improved_predictions.png")
    
    return model, history, val_ids, carbon_report, sle_report

# ==================== Main Entry Point ====================

if __name__ == "__main__":
    # Configuration (same as before)
    config = {
        # Data paths
        'image_dir': 'data/images',
        'bleached_mask_dir': 'data/masks_bleached',
        'non_bleached_mask_dir': 'data/masks_non_bleached',
        
        # Model
        'backbone': 'tf_efficientnet_b2_ns',
        
        # Training
        'batch_size': 4,
        'epochs': 60,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'img_size': 512,
        'num_workers': 4,
        'seed': 42,
        
        # Loss weights
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
    
    print("="*70)
    print("[START] TRAINING CORAL BLEACHING MODEL WITH CARBON TRACKING")
    print("="*70)
    print("\nThis enhanced training pipeline will:")
    print("[CHECK] Track carbon emissions during training")
    print("[CHECK] Measure inference efficiency")
    print("[CHECK] Calculate model complexity metrics")
    print("[CHECK] Generate break-even analysis for your SLE essay")
    print("[CHECK] Provide optimization recommendations")
    print("\n" + "="*70 + "\n")
    
    # Run training with carbon tracking
    model, history, val_ids, carbon_report, sle_report = train_with_carbon_tracking(config)
    
    print("\n" + "="*70)
    print("[DONE] TRAINING COMPLETE!")
    print("="*70)
    print("\nYour SLE essay now has quantitative evidence for:")
    print("  [CHECK] Question 1: Computational resource footprint")
    print("  [CHECK] Question 3: When costs outweigh benefits")
    print("  [CHECK] Question 4: Practices to reduce environmental footprint")
    print("\nAll reports and visualizations have been saved.")
    print("="*70)