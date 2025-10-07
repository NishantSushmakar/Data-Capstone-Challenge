# Data-Capstone-Challenge

This project implements coral bleaching detection and automated quantification of carbon emissions using deep learning, with a focus on environmental sustainability and computational efficiency. The system performs multi-task learning including:

- **Semantic Segmentation**: Classifies each pixel as background, healthy coral, or bleached coral
- **Bleaching Ratio Prediction**: Estimates the percentage of bleached coral in an image
- **Coral Coverage Estimation**: Calculates total coral coverage in the scene



## Model Comparison

### Architecture & Configuration

| **Component**              | **Baseline Model**         | **Improved Model**                | **Improvement**          |
| -------------------------- | -------------------------- | --------------------------------- | ------------------------ |
| **Backbone**               | EfficientNet-B0            | EfficientNet-B2 (Noisy Student)   | +2 scale, better weights |
| **Parameters**             | ~5M                        | ~12\.4M                           | +148% capacity           |
| **Multi-scale Features**   | Basic U-Net decoder        | ASPP + Skip connections           | Enhanced context         |
| **Attention Mechanism**    | None                       | CBAM (Channel + Spatial)          | Feature refinement       |
| **Segmentation Loss**      | Cross-Entropy              | Focal Loss + Dice Loss            | Class imbalance handling |
| **Regression Loss**        | MSE                        | Huber Loss                        | Outlier robustness       |
| **Data Augmentation**      | Basic (flip, rotate, blur) | Enhanced underwater-specific      | Domain adaptation        |
| **Learning Rate Strategy** | Single LR                  | Differential (0.1×/0.5×/1×)       | Better convergence       |
| **Data Quality Control**   | No filtering               | Filter samples <5% coral coverage | Reduces noise            |
| **Batch Size**             | 8                          | 4                                 | Memory optimization      |
| **Training Epochs**        | 50                         | 60                                | Extended training        |
| **Image Resolution**       | 512×512                    | 512×512                           | Consistent               |

### Performance Metrics

| **Metric**                    | **Baseline Model** | **Improved Model** | **Change** |
| ----------------------------- | ------------------ | ------------------ | ---------- |
| **Validation Loss**           | 0.6700             | 0.9982             | +48.9%     |
| **Mean IoU**                  | 0.4965             | 0.5432             | **+9.4%**  |
| **Bleaching MAE**             | 0.1596             | 0.1314             | **-21.5%** |
| **Bleaching Accuracy (±10%)** | 0.4725             | 0.6125             | **+29.6%** |
| **Model Size**                | ~20MB              | ~35MB              | +75%       |

**Note**: The higher validation loss in the improved model is due to the different loss function composition (Focal + Dice + Huber vs. CE + MSE). When comparing task-specific metrics like Bleaching MAE, the improved model shows better performance.



## Carbon Footprint & Environmental Impact

### Why Track Carbon Emissions?

This project includes comprehensive carbon emission tracking to address the ethical question: **"Do the environmental costs of training AI models justify their benefits in coral reef conservation?"**

### Tools & Methodology

We use **CodeCarbon** (industry-standard carbon tracking) and **THOP** (model efficiency analysis) to quantify:

1. **Training Phase Emissions**: One-time cost of model training
2. **Inference Phase Emissions**: Per-image prediction cost
3. **Model Efficiency**: FLOPs, parameters, latency
4. **Break-Even Analysis**: When AI becomes carbon-positive vs traditional monitoring

```bash
# Install carbon tracking dependencies
pip install codecarbon thop

# Train with automatic carbon tracking
python train_with_carbon_tracking.py

# Generated reports:
# - coral_bleaching_carbon_report.json       (raw emissions data)
# - sle_environmental_impact_report.json     (structured analysis)
# - sle_environmental_analysis.png           (visualizations)
```

Training Emissions: 0.1775 kg CO2eq
Training Duration: 2.01 hours
Model Parameters: 12.41M
FLOPs: 6.13 GFLOPs

Equivalent to:
   Driving 0.8 km in a car
   0.01 trees needed for 1 year to offset
   22 smartphone charges