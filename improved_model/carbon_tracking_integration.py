"""
Carbon Emission Tracking for Coral Bleaching Model
Integrates CodeCarbon to quantify environmental costs for SLE analysis

Key Metrics Tracked:
1. Training phase carbon emissions (kg CO2eq)
2. Inference phase energy consumption (with proper warmup and batching)
3. Model efficiency metrics (FLOPs, params, latency)
4. Cost-benefit analysis framework

Installation required:
pip install codecarbon thop

USAGE:
For complete workflow, use training_with_carbon_tracking.py which integrates
all fixes and generates comprehensive SLE reports.
"""

import torch
import torch.nn as nn
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
import time
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
try:
    from thop import profile  # pip install thop
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not installed. FLOPs calculation will be skipped.")
import matplotlib.pyplot as plt

# ==================== Carbon Tracking Integration ====================

class CarbonAwareTrainer:
    """Wrapper to track carbon emissions during training"""
    
    def __init__(self, 
                 project_name: str = "coral_bleaching_detection",
                 country_iso_code: str = "NLD",  # Netherlands
                 region: str = "North Holland",
                 tracking_mode: str = "online",  # or "offline"
                 output_dir: str = "./emissions"):
        """
        Initialize carbon tracker
        
        Args:
            project_name: Name of your project
            country_iso_code: ISO code for your location (for grid carbon intensity)
            region: Your region (optional, for more precise carbon intensity)
            tracking_mode: "online" or "offline" 
                - online: automatic API calls for real-time carbon intensity
                - offline: uses regional averages
            output_dir: Where to save emission logs
        """
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize tracker based on mode
        if tracking_mode == "online":
            self.tracker = EmissionsTracker(
                project_name=project_name,
                output_dir=str(self.output_dir),
                country_iso_code=country_iso_code,
                region=region,
                log_level="warning"
            )
        else:
            self.tracker = OfflineEmissionsTracker(
                project_name=project_name,
                output_dir=str(self.output_dir),
                country_iso_code=country_iso_code,
                log_level="warning"
            )
        
        self.training_emissions = []
        self.inference_emissions = []
    
    def start_training(self):
        """Start tracking training emissions"""
        print(f"\n{'='*60}")
        print(f"[EARTH] Starting carbon emission tracking for training")
        print(f"{'='*60}\n")
        self.tracker.start()
        self.training_start_time = time.time()
    
    def stop_training(self):
        """Stop tracking and save training emissions"""
        training_time = time.time() - self.training_start_time
        emissions = self.tracker.stop()
        
        self.training_emissions.append({
            'phase': 'training',
            'emissions_kg_co2eq': emissions,
            'duration_hours': training_time / 3600
        })
        
        print(f"\n{'='*60}")
        print(f"[EARTH] Training Carbon Footprint")
        print(f"{'='*60}")
        print(f"Total Emissions: {emissions:.4f} kg CO2eq")
        print(f"Training Time: {training_time/3600:.2f} hours")
        print(f"Emission Rate: {emissions/(training_time/3600):.4f} kg CO2eq/hour")
        print(f"{'='*60}\n")
        
        return emissions
    
    def track_inference(self, model, dataloader, device, num_samples=None):
        """
        Track emissions for inference/deployment simulation

        FIXED: Proper warmup, GPU sync, and batch processing
        """
        print(f"\n{'='*60}")
        print(f"[EARTH] Tracking inference emissions (with proper warmup)")
        print(f"{'='*60}\n")

        model.eval()
        model.to(device)

        # Verify setup
        print(f"Device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        sample_batch = next(iter(dataloader))
        batch_size = sample_batch['image'].shape[0]
        print(f"Batch size: {batch_size}")
        if batch_size == 1:
            print("⚠️  WARNING: batch_size=1 is inefficient! Use batch_size=4-8\n")
        else:
            print(f"✓ Using batch processing\n")

        # CRITICAL: Warmup phase (excluded from measurement)
        print("Running warmup...")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 10:  # 10 warmup batches
                    break
                images = batch['image'].to(device)
                _ = model(images)
                if i == 0 and device.type == 'cuda':
                    torch.cuda.synchronize()
        print("✓ Warmup complete\n")

        # CRITICAL FIX: Create a new tracker instance for inference
        # (the main tracker was already stopped after training)
        try:
            # Try new API first (codecarbon >= 2.0)
            inference_tracker = EmissionsTracker(
                project_name=f"{self.project_name}_inference",
                output_dir=str(self.output_dir),
                country_iso_code="NLD",
                log_level="warning"
            )
        except TypeError:
            # Fall back to old API (codecarbon < 2.0)
            inference_tracker = EmissionsTracker(
                project_name=f"{self.project_name}_inference",
                output_dir=str(self.output_dir),
                log_level="warning"
            )

        # Start tracking AFTER warmup
        inference_tracker.start()
        start_time = time.time()

        predictions = 0
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if num_samples and predictions >= num_samples:
                    break

                images = batch['image'].to(device)
                _ = model(images)
                predictions += images.size(0)

        # CRITICAL: Ensure GPU finishes all work
        if device.type == 'cuda':
            torch.cuda.synchronize()

        inference_time = time.time() - start_time
        emissions = inference_tracker.stop()

        # Safety check for None emissions
        if emissions is None:
            print("⚠️  WARNING: CodeCarbon returned None emissions. Using fallback calculation.")
            # Fallback: estimate based on GPU power and time
            # RTX 2060 TDP ~160W, typical inference load ~70%
            gpu_power_kw = 0.160 * 0.7  # 112W in kW
            emissions = (gpu_power_kw * inference_time / 3600) * 0.5  # ~0.5 kg CO2/kWh for NL grid
            print(f"   Estimated emissions: {emissions:.6f} kg CO2eq")

        # Calculate per-prediction metrics
        time_per_prediction = inference_time / predictions
        emissions_per_prediction = emissions / predictions

        self.inference_emissions.append({
            'phase': 'inference',
            'total_emissions_kg_co2eq': emissions,
            'num_predictions': predictions,
            'emissions_per_prediction_g': emissions_per_prediction * 1000,
            'time_per_prediction_ms': time_per_prediction * 1000,
            'throughput_images_per_sec': predictions / inference_time
        })

        print(f"\n{'='*60}")
        print(f"[EARTH] Inference Carbon Footprint")
        print(f"{'='*60}")
        print(f"Total Emissions: {emissions:.6f} kg CO2eq")
        print(f"Predictions Made: {predictions}")
        print(f"Per-Prediction: {emissions_per_prediction*1000:.6f} g CO2eq")
        print(f"Inference Time: {time_per_prediction*1000:.2f} ms per image")
        print(f"Throughput: {predictions/inference_time:.1f} images/sec")
        print(f"{'='*60}\n")

        return emissions
    
    def save_detailed_report(self, filename="carbon_report.json"):
        """Save comprehensive carbon tracking report"""
        report = {
            'project': self.project_name,
            'training': self.training_emissions,
            'inference': self.inference_emissions,
            'summary': self._generate_summary()
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[OK] Detailed carbon report saved to: {output_path}")
        return report
    
    def _generate_summary(self):
        """Generate summary statistics"""
        if not self.training_emissions:
            return {}
        
        total_training_emissions = sum(e['emissions_kg_co2eq'] 
                                      for e in self.training_emissions)
        
        summary = {
            'total_training_emissions_kg_co2eq': total_training_emissions,
            'training_duration_hours': sum(e['duration_hours'] 
                                          for e in self.training_emissions)
        }
        
        if self.inference_emissions:
            latest_inference = self.inference_emissions[-1]
            summary.update({
                'inference_emissions_per_prediction_g': 
                    latest_inference['emissions_per_prediction_g'],
                'inference_time_per_prediction_ms': 
                    latest_inference['time_per_prediction_ms']
            })
        
        return summary

# ==================== Model Efficiency Metrics ====================

class ModelEfficiencyAnalyzer:
    """Analyze model computational efficiency"""
    
    @staticmethod
    def calculate_flops(model, input_size=(1, 3, 512, 512)):
        """Calculate FLOPs (floating point operations)"""
        if not THOP_AVAILABLE:
            print("Warning: thop not available, skipping FLOPs calculation")
            params = sum(p.numel() for p in model.parameters())
            return {
                'flops': 0,
                'flops_billions': 0,
                'params': params,
                'params_millions': params / 1e6
            }
        
        try:
            device = next(model.parameters()).device
            input_tensor = torch.randn(input_size).to(device)
            
            flops, params = profile(model, inputs=(input_tensor,), verbose=False)
            
            return {
                'flops': flops,
                'flops_billions': flops / 1e9,
                'params': params,
                'params_millions': params / 1e6
            }
        except Exception as e:
            print(f"Warning: Could not calculate FLOPs: {e}")
            params = sum(p.numel() for p in model.parameters())
            return {
                'flops': 0,
                'flops_billions': 0,
                'params': params,
                'params_millions': params / 1e6
            }
    
    @staticmethod
    def measure_latency(model, input_size=(1, 3, 512, 512), 
                       device='cuda', num_iterations=100):
        """Measure inference latency"""
        model.eval()
        input_tensor = torch.randn(input_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Measure
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_tensor)
        
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_latency = (end_time - start_time) / num_iterations
        
        return {
            'avg_latency_ms': avg_latency * 1000,
            'throughput_fps': 1.0 / avg_latency
        }
    
    @staticmethod
    def calculate_model_size(model, filepath=None):
        """Calculate model size in MB"""
        if filepath:
            size_mb = Path(filepath).stat().st_size / (1024 * 1024)
        else:
            # Calculate from parameters
            param_size = sum(p.nelement() * p.element_size() 
                           for p in model.parameters())
            buffer_size = sum(b.nelement() * b.element_size() 
                            for b in model.buffers())
            size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        return {'model_size_mb': size_mb}
    
    @staticmethod
    def comprehensive_analysis(model, input_size=(1, 3, 512, 512), 
                              device='cuda', model_path=None):
        """Run comprehensive efficiency analysis"""
        print(f"\n{'='*60}")
        print(f"[CHART] Model Efficiency Analysis")
        print(f"{'='*60}\n")
        
        results = {}
        
        # FLOPs and parameters
        flops_info = ModelEfficiencyAnalyzer.calculate_flops(model, input_size)
        results.update(flops_info)
        print(f"FLOPs: {flops_info['flops_billions']:.2f} GFLOPs")
        print(f"Parameters: {flops_info['params_millions']:.2f}M")
        
        # Latency
        latency_info = ModelEfficiencyAnalyzer.measure_latency(
            model, input_size, device
        )
        results.update(latency_info)
        print(f"Inference Latency: {latency_info['avg_latency_ms']:.2f} ms")
        print(f"Throughput: {latency_info['throughput_fps']:.1f} FPS")
        
        # Model size
        size_info = ModelEfficiencyAnalyzer.calculate_model_size(model, model_path)
        results.update(size_info)
        print(f"Model Size: {size_info['model_size_mb']:.2f} MB")
        
        # Efficiency score (lower is better)
        if flops_info['flops_billions'] > 0:
            efficiency_score = (
                flops_info['flops_billions'] * 
                latency_info['avg_latency_ms'] * 
                size_info['model_size_mb']
            ) / 1000
        else:
            efficiency_score = (
                latency_info['avg_latency_ms'] * 
                size_info['model_size_mb']
            ) / 100
        results['efficiency_score'] = efficiency_score
        
        print(f"\nEfficiency Score: {efficiency_score:.2f}")
        print(f"  (lower is better, combines FLOPs, latency, size)")
        print(f"{'='*60}\n")
        
        return results

# ==================== Cost-Benefit Analysis Framework ====================

class EnvironmentalImpactAnalyzer:
    """Framework for SLE cost-benefit analysis"""
    
    def __init__(self):
        # Reference values for context
        self.references = {
            'car_km_per_kg_co2': 4.6,  # km driven per kg CO2
            'tree_year_kg_co2': 21,     # kg CO2 absorbed by tree per year
            'smartphone_charge_kg_co2': 0.008,  # kg CO2 per charge
            'reef_survey_boat_kg_co2_per_hour': 15,  # Traditional boat survey
            'human_breath_kg_co2_per_day': 1.0
        }
    
    def compare_to_references(self, emissions_kg):
        """Convert emissions to relatable equivalents"""
        return {
            'equivalent_km_driven': emissions_kg * self.references['car_km_per_kg_co2'],
            'trees_year_to_offset': emissions_kg / self.references['tree_year_kg_co2'],
            'smartphone_charges': emissions_kg / self.references['smartphone_charge_kg_co2'],
            'days_of_human_breathing': emissions_kg / self.references['human_breath_kg_co2_per_day']
        }
    
    def calculate_break_even_point(self, 
                                   training_emissions_kg,
                                   inference_emissions_per_image_g,
                                   traditional_survey_emissions_kg_per_survey,
                                   images_per_traditional_survey=1000):
        """
        Calculate when AI model breaks even with traditional methods
        
        Args:
            training_emissions_kg: One-time training cost
            inference_emissions_per_image_g: Per-prediction cost
            traditional_survey_emissions_kg_per_survey: Traditional method cost
            images_per_traditional_survey: Images analyzed per survey
        """
        
        # Convert to same units
        inference_per_image_kg = inference_emissions_per_image_g / 1000
        
        # Traditional method per image
        traditional_per_image_kg = (traditional_survey_emissions_kg_per_survey / 
                                   images_per_traditional_survey)
        
        # If AI is more expensive per image, never breaks even
        if inference_per_image_kg >= traditional_per_image_kg:
            return {
                'break_even_images': float('inf'),
                'verdict': 'AI model never breaks even - inference too expensive',
                'ai_emissions_per_image_kg': inference_per_image_kg,
                'traditional_emissions_per_image_kg': traditional_per_image_kg
            }
        
        # Calculate break-even
        savings_per_image = traditional_per_image_kg - inference_per_image_kg
        break_even_images = training_emissions_kg / savings_per_image
        
        return {
            'break_even_images': break_even_images,
            'break_even_surveys': break_even_images / images_per_traditional_survey,
            'ai_training_cost_kg': training_emissions_kg,
            'ai_emissions_per_image_kg': inference_per_image_kg,
            'traditional_emissions_per_image_kg': traditional_per_image_kg,
            'savings_per_image_kg': savings_per_image,
            'verdict': f'Breaks even after {break_even_images:.0f} images '
                      f'({break_even_images/images_per_traditional_survey:.1f} surveys)'
        }
    
    def generate_sle_report(self, carbon_report, efficiency_metrics, 
                           traditional_method_assumptions):
        """Generate comprehensive report for SLE essay"""
        
        training_emissions = carbon_report['summary']['total_training_emissions_kg_co2eq']
        inference_per_pred = carbon_report['summary']['inference_emissions_per_prediction_g']
        
        report = {
            'question_1_computational_footprint': {
                'training_phase': {
                    'total_emissions_kg_co2eq': training_emissions,
                    'duration_hours': carbon_report['summary']['training_duration_hours'],
                    'equivalent_comparisons': self.compare_to_references(training_emissions),
                    'model_complexity': {
                        'parameters_millions': efficiency_metrics['params_millions'],
                        'flops_billions': efficiency_metrics['flops_billions'],
                        'model_size_mb': efficiency_metrics['model_size_mb']
                    }
                },
                'deployment_phase': {
                    'emissions_per_prediction_g': inference_per_pred,
                    'latency_ms': efficiency_metrics['avg_latency_ms'],
                    'throughput_fps': efficiency_metrics['throughput_fps']
                }
            },
            
            'question_3_cost_benefit_analysis': self.calculate_break_even_point(
                training_emissions,
                inference_per_pred,
                traditional_method_assumptions['emissions_kg_per_survey'],
                traditional_method_assumptions['images_per_survey']
            ),
            
            'question_4_optimization_recommendations': self._generate_recommendations(
                efficiency_metrics, training_emissions
            )
        }
        
        return report
    
    def _generate_recommendations(self, efficiency_metrics, training_emissions):
        """Generate practical recommendations for reducing footprint"""
        recommendations = []
        
        # Model size recommendations
        if efficiency_metrics['params_millions'] > 30:
            recommendations.append({
                'category': 'Model Architecture',
                'issue': f"Large model ({efficiency_metrics['params_millions']:.1f}M params)",
                'suggestion': 'Consider knowledge distillation or pruning',
                'potential_reduction': '30-50% parameter reduction possible'
            })
        
        # Training recommendations
        if training_emissions > 1.0:  # > 1 kg CO2
            recommendations.append({
                'category': 'Training Strategy',
                'issue': f'High training emissions ({training_emissions:.2f} kg CO2)',
                'suggestion': 'Use transfer learning, reduce epochs, or train on renewable energy grids',
                'potential_reduction': '50-70% with aggressive transfer learning'
            })
        
        # Inference recommendations
        if efficiency_metrics['avg_latency_ms'] > 50:
            recommendations.append({
                'category': 'Inference Optimization',
                'issue': f"Slow inference ({efficiency_metrics['avg_latency_ms']:.1f} ms)",
                'suggestion': 'Model quantization (INT8), TensorRT optimization, or edge deployment',
                'potential_reduction': '2-4x speedup, lower energy per prediction'
            })
        
        return recommendations
    
    def visualize_analysis(self, carbon_report, break_even_analysis, save_path='sle_analysis.png'):
        """Create visualization for SLE essay"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Training emissions breakdown
        training_data = carbon_report['training'][0]
        ax1 = axes[0, 0]
        comparisons = self.compare_to_references(
            training_data['emissions_kg_co2eq']
        )
        labels = ['Training\nEmissions', 'Car km\nequivalent', 
                 'Smartphone\ncharges', 'Days of\nbreathing']
        values = [
            training_data['emissions_kg_co2eq'],
            comparisons['equivalent_km_driven'] / 100,  # Scale down
            comparisons['smartphone_charges'] / 1000,    # Scale down
            comparisons['days_of_human_breathing']
        ]
        ax1.bar(labels, values, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
        ax1.set_ylabel('Scaled Values')
        ax1.set_title('Training Emissions in Context', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Break-even analysis
        ax2 = axes[0, 1]
        if break_even_analysis['break_even_images'] != float('inf'):
            images = np.linspace(0, break_even_analysis['break_even_images'] * 2, 100)
            ai_cost = (break_even_analysis['ai_training_cost_kg'] + 
                      images * break_even_analysis['ai_emissions_per_image_kg'])
            trad_cost = images * break_even_analysis['traditional_emissions_per_image_kg']
            
            ax2.plot(images, ai_cost, label='AI System', linewidth=2, color='#3498db')
            ax2.plot(images, trad_cost, label='Traditional Method', 
                    linewidth=2, color='#e74c3c')
            ax2.axvline(break_even_analysis['break_even_images'], 
                       color='green', linestyle='--', linewidth=2, 
                       label=f"Break-even: {break_even_analysis['break_even_images']:.0f} images")
            ax2.set_xlabel('Number of Images Analyzed')
            ax2.set_ylabel('Cumulative CO2 Emissions (kg)')
            ax2.set_title('Cost-Benefit Break-Even Analysis', 
                         fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'AI model more expensive\nper prediction', 
                    ha='center', va='center', fontsize=12)
            ax2.set_title('No Break-Even Point', fontsize=12, fontweight='bold')
        
        # 3. Deployment efficiency
        ax3 = axes[1, 0]
        if carbon_report['inference']:
            inference = carbon_report['inference'][0]
            metrics = ['Emissions\n(g CO2/pred)', 'Latency\n(ms)', 
                      'Throughput\n(imgs/sec)']
            values = [
                inference['emissions_per_prediction_g'] * 100,  # Scale for visibility
                inference['time_per_prediction_ms'] / 10,        # Scale
                inference['throughput_images_per_sec']
            ]
            ax3.bar(metrics, values, color=['#e74c3c', '#f39c12', '#2ecc71'])
            ax3.set_ylabel('Scaled Values')
            ax3.set_title('Deployment Efficiency Metrics', 
                         fontsize=12, fontweight='bold')
        
        # 4. Lifecycle emissions
        ax4 = axes[1, 1]
        training_total = carbon_report['summary']['total_training_emissions_kg_co2eq']
        
        # Simulate deployment over 1 year (assume 1000 images/day)
        daily_images = 1000
        days_per_year = 365
        yearly_inference = (daily_images * days_per_year * 
                           inference['emissions_per_prediction_g'] / 1000)
        
        phases = ['Training\n(one-time)', 'Inference\n(1 year)', 'Total']
        emissions = [training_total, yearly_inference, training_total + yearly_inference]
        colors = ['#3498db', '#e67e22', '#e74c3c']
        
        bars = ax4.bar(phases, emissions, color=colors)
        ax4.set_ylabel('CO2 Emissions (kg)')
        ax4.set_title('Lifecycle Carbon Footprint', fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f} kg',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n[OK] SLE analysis visualization saved to: {save_path}")
        plt.close()

# ==================== Example Integration ====================

def example_usage():
    """
    Example of how to integrate carbon tracking into your training

    NOTE: This is a template. For actual use, see training_with_carbon_tracking.py
    """

    print("="*60)
    print("Carbon Tracking Integration Template")
    print("="*60)
    print("\nIntegration steps:")
    print("1. Initialize: carbon_tracker = CarbonAwareTrainer(...)")
    print("2. Before training: carbon_tracker.start_training()")
    print("3. After training: carbon_tracker.stop_training()")
    print("4. Analyze model: ModelEfficiencyAnalyzer.comprehensive_analysis(model, ...)")
    print("5. Track inference: carbon_tracker.track_inference(model, dataloader, ...)")
    print("6. Generate reports: impact_analyzer.generate_sle_report(...)")
    print("\nFor complete working example, run:")
    print("  python training_with_carbon_tracking.py")
    print("="*60)

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    print("""
1. Install requirements:
   pip install codecarbon thop

2. Import in your improved_model.py:
   from carbon_tracking_integration import CarbonAwareTrainer, ModelEfficiencyAnalyzer, EnvironmentalImpactAnalyzer

3. Wrap your training:
   carbon_tracker = CarbonAwareTrainer(country_iso_code="NLD")
   carbon_tracker.start_training()
   # ... train your model ...
   carbon_tracker.stop_training()

4. Analyze efficiency:
   efficiency = ModelEfficiencyAnalyzer.comprehensive_analysis(model)

5. Generate SLE report:
   impact = EnvironmentalImpactAnalyzer()
   sle_report = impact.generate_sle_report(...)

See example_usage() function for complete workflow.
    """)