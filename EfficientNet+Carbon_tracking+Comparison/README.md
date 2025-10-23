# Coral Bleaching Detection with Carbon Footprint Analysis

## Overview

This implements an AI-powered coral bleaching detection system using deep learning, with integrated carbon emission tracking to evaluate environmental sustainability. The system addresses coral reef conservation through automated image analysis while maintaining awareness of its computational environmental costs.

**Key Features:**
- **Semantic Segmentation**: Pixel-wise classification (background, healthy coral, bleached coral)
- **Bleaching Ratio Prediction**: Automated estimation of bleaching percentage
- **Coral Coverage Estimation**: Quantifies total coral presence in images
- **Carbon Footprint Tracking**: Real-time monitoring of training and inference emissions
- **Environmental Impact Analysis**: Break-even analysis comparing AI vs traditional monitoring methods

**Programming Language**: Python 3.9+
**Framework**: PyTorch 1.10+
**Platform**: Tested on Windows 10/11 with CUDA-enabled GPU (NVIDIA RTX 2060 or higher recommended)

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Model Architecture](#model-architecture)
5. [Performance Metrics](#performance-metrics)
6. [Carbon Footprint Analysis](#carbon-footprint-analysis)
7. [Troubleshooting](#troubleshooting)
8. [Credits and Acknowledgements](#credits-and-acknowledgements)
9. [Ethical Considerations](#ethical-considerations)

---

## Project Structure

```
Data-Capstone-Challenge/
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies with versions
│
├── baseline_model/                        # Baseline model implementation
│   ├── model.py                          # Baseline model architecture
│   ├── training_history.png              # Training curves
│   └── predictions_visualization.png     # Sample predictions
│
├── improved_model/                        # Enhanced model with carbon tracking
│   ├── improved_model.py                 # Main model architecture and training
│   ├── carbon_tracking_integration.py    # Carbon emission tracking utilities
│   ├── training_with_carbon_tracking.py  # Training script with carbon analysis
│   ├── improved_training_history.png     # Training performance graphs
│   ├── improved_predictions.png          # Model prediction visualizations
│
├── model_comparison/                      # Architecture comparison experiments
│   ├── compare_architectures.py          # EfficientNet vs ResNet comparison
│   ├── architecture_comparison.csv       # Quantitative comparison results
│   └── detailed_comparison_results.json  # Detailed metrics in JSON format
│
├── SLE/                                   # Environmental impact analysis (for SLE essay)
│   ├── sle_environmental_impact_report.json  # Complete carbon analysis report
│   ├── sle_environmental_analysis.png        # Break-even analysis visualization
│   └── CARBON_RESULTS_README.md              # Guide to using carbon tracking results
│
├── results/                               # Data augmentation and Model analysis
│   ├── image_transformations_demo.ipynb       # Data augmentation demonstration
│   ├── model_architecture_analysis.ipynb      # Model structure analysis notebook
│   └── all.png                                # Project overview diagram
```

---

## Installation

### Prerequisites

- **Python**: Version 3.9 or higher
- **GPU**: NVIDIA GPU with CUDA support (optional but highly recommended)
- **CUDA**: Version 11.3 or higher (if using GPU)
- **Storage**: At least 2GB free disk space for models and dependencies

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Data-Capstone-Challenge
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

The project includes a `requirements.txt` file with all necessary packages and their tested versions:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch>=1.10.0` - Deep learning framework
- `torchvision>=0.11.0` - Computer vision utilities
- `timm>=0.6.0` - Pre-trained model library
- `opencv-python>=4.5.0` - Image processing
- `albumentations>=1.3.0` - Advanced data augmentation
- `codecarbon>=2.0.0` - Carbon emission tracking
- `thop>=0.1.0` - Model efficiency analysis (FLOPs calculation)
- `matplotlib>=3.5.0` - Visualization
- `scikit-learn>=1.0.0` - ML utilities

**If installation fails**, you can install packages individually:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm opencv-python albumentations codecarbon thop matplotlib scikit-learn
```

### Step 4: Verify Installation

Run this command to verify your setup:

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 1.10.0 (or higher)
CUDA Available: True
```

---

## Usage

### Dataset Preparation

The model expects coral reef images with corresponding segmentation masks. Organize your data as follows:

```
data/
├── images/              # RGB images (.jpg or .JPG)
├── masks_bleached/      # Binary masks for bleached coral (.png)
└── masks_non_bleached/  # Binary masks for healthy coral (.png)
```

**Image Requirements:**
- Format: JPEG (.jpg or .JPG, case-insensitive)
- Resolution: Any (automatically resized to 512×512)
- Naming: Image and mask files must share the same base name (e.g., `img001.jpg`, `img001.png`)

### Training the Baseline Model

```bash
cd baseline_model
python model.py
```

**Configuration** (edit in `model.py`):
```python
config = {
    'image_dir': 'data/images',
    'bleached_mask_dir': 'data/masks_bleached',
    'non_bleached_mask_dir': 'data/masks_non_bleached',
    'batch_size': 8,
    'epochs': 50,
    'lr': 1e-4,
    'img_size': 512
}
```

**Outputs:**
- `best_coral_model.pth` - Best model checkpoint
- `training_history.png` - Loss and metric curves
- `predictions_visualization.png` - Sample predictions on validation set

### Training the Improved Model (with Carbon Tracking)

```bash
cd improved_model
python training_with_carbon_tracking.py
```

This script performs:
1. ✅ Model training with early stopping
2. ✅ Real-time carbon emission tracking
3. ✅ Model efficiency analysis (FLOPs, parameters, latency)
4. ✅ Inference cost measurement
5. ✅ Break-even analysis (AI vs traditional methods)
6. ✅ Comprehensive report generation

**Outputs:**
- `best_improved_model.pth` - Best model by validation loss
- `best_iou_model.pth` - Best model by IoU metric
- `coral_bleaching_carbon_report.json` - Raw carbon tracking data
- `sle_environmental_impact_report.json` - SLE analysis with recommendations
- `dataset_split.json` - Train/validation split for reproducibility
- `sle_environmental_analysis.png` - Carbon footprint visualizations
- `improved_training_history.png` - Training curves
- `improved_predictions.png` - Model predictions
- `CARBON_RESULTS_README.md` - Detailed guide for interpreting results

### Running Inference on New Images

```python
import torch
from improved_model.improved_model import ImprovedCoralModel
from PIL import Image
import torchvision.transforms as T

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedCoralModel(backbone_name='tf_efficientnet_b2_ns', num_classes=3)
checkpoint = torch.load('improved_model/best_improved_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

# Prepare image
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = Image.open('your_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    seg_output, bleaching_pred, coverage_pred = model(input_tensor)

print(f"Bleaching Percentage: {bleaching_pred.item()*100:.2f}%")
print(f"Coral Coverage: {coverage_pred.item()*100:.2f}%")
```

### Comparing Model Architectures

To compare different backbone architectures (EfficientNet vs ResNet):

```bash
cd model_comparison
python compare_architectures.py
```

**Outputs:**
- `architecture_comparison.csv` - Performance comparison table
- `detailed_comparison_results.json` - Full metrics in JSON format
- Trained checkpoints for each architecture

---

## Model Architecture

### Baseline Model
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Decoder**: Simple U-Net style decoder
- **Loss**: Cross-Entropy (segmentation) + MSE (regression)
- **Parameters**: ~5M

### Improved Model
- **Backbone**: EfficientNet-B2 with Noisy Student weights (better initialization)
- **Multi-scale Features**: Atrous Spatial Pyramid Pooling (ASPP)
- **Attention**: Convolutional Block Attention Module (CBAM)
- **Loss Functions**:
  - Focal Loss + Dice Loss (segmentation) - handles class imbalance
  - Huber Loss (regression) - robust to outliers
- **Differential Learning Rates**:
  - Backbone: 0.1× base LR
  - Decoder: 0.5× base LR
  - Prediction heads: 1× base LR
- **Data Quality Control**: Filters training samples with <5% coral coverage
- **Parameters**: ~12.4M

**Architecture Diagram**: See `all.png` for visual overview

---

## Performance Metrics

### Quantitative Results

| **Metric**                    | **Baseline Model** | **Improved Model** | **Improvement** |
| ----------------------------- | ------------------ | ------------------ | --------------- |
| **Mean IoU**                  | 0.4965             | 0.5432             | **+9.4%**       |
| **Bleaching MAE**             | 0.1596             | 0.1314             | **-21.5%**      |
| **Bleaching Accuracy (±10%)** | 47.25%             | 61.25%             | **+29.6%**      |
| **Validation Loss**           | 0.6700             | 0.9982             | N/A*            |
| **Model Size**                | ~20MB              | ~35MB              | +75%            |

*The validation loss is not directly comparable due to different loss function compositions.

### Key Improvements
1. **Better Segmentation**: IoU improved by 9.4% through ASPP and CBAM attention
2. **More Accurate Bleaching Estimation**: MAE reduced by 21.5% using Huber loss
3. **Higher Reliability**: 61% of predictions within ±10% error margin (vs 47% baseline)

---

## Carbon Footprint Analysis

### Training Phase

- **Total Emissions**: ~0.XX kg CO2eq (varies by hardware and training duration)
- **Training Time**: ~X hours on NVIDIA RTX 2060
- **Equivalent Impact**: Driving ~X km in a car

### Inference Phase (Per Image)

- **Emissions**: ~X.XXXXXX g CO2eq per prediction
- **Latency**: ~XX ms per image (batch size = 8)
- **Throughput**: ~XX images/second

### Break-Even Analysis

The improved model becomes carbon-positive compared to traditional human-based coral surveys after analyzing **~X,XXX images** (approximately X traditional surveys).

**Key Findings**:
- ✅ AI inference is XX× more carbon-efficient than manual surveys per image
- ✅ One-time training cost is amortized over thousands of predictions
- ✅ Model can process entire reef surveys in minutes vs days

**For detailed analysis**: See `SLE/sle_environmental_impact_report.json` and `SLE/CARBON_RESULTS_README.md`

---

## Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory Error**

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size in config: `'batch_size': 2` (instead of 4)
- Close other GPU-using applications
- Use CPU instead: Set `device = torch.device('cpu')` in the script

#### 2. **CodeCarbon Returns None Emissions**

**Error**: `WARNING: CodeCarbon returned None emissions`

**Solutions**:
- This is handled automatically with fallback estimation
- Update CodeCarbon: `pip install --upgrade codecarbon`
- Check network connection (for online tracking mode)

#### 3. **File Not Found: .JPG vs .jpg**

**Error**: `FileNotFoundError: No such file or directory`

**Solutions**:
- The improved model handles both `.jpg` and `.JPG` (case-insensitive)
- If using baseline model, ensure all images use lowercase `.jpg`
- Or modify line 38 in `model.py`: `if f.lower().endswith('.jpg')`

#### 4. **Import Error: No module named 'thop'**

**Error**: `ModuleNotFoundError: No module named 'thop'`

**Solutions**:
```bash
pip install thop
```

#### 5. **Slow Training on CPU**

If you don't have a GPU, training will be significantly slower (hours → days).

**Solutions**:
- Use Google Colab with free GPU: [colab.research.google.com](https://colab.research.google.com)
- Reduce `epochs` to 20-30 for faster experiments
- Use pre-trained checkpoints for inference only

#### 6. **Different Results After Retraining**

**Cause**: Random seed not set or different hardware

**Solution**: We save `dataset_split.json` with train/val split. Use this for reproducibility:
```python
import json
with open('dataset_split.json', 'r') as f:
    split = json.load(f)
train_ids = split['train_ids']
val_ids = split['val_ids']
```

---

## Credits and Acknowledgements

### External Libraries and Resources

This project builds upon the following open-source tools and libraries:

1. **PyTorch** - Deep learning framework
   [https://pytorch.org/](https://pytorch.org/)

2. **timm (PyTorch Image Models)** - Pre-trained model library
   [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)

3. **Albumentations** - Advanced data augmentation library
   [https://albumentations.ai/](https://albumentations.ai/)

4. **CodeCarbon** - Carbon emission tracking for ML
   [https://codecarbon.io/](https://codecarbon.io/)

5. **THOP** - PyTorch model FLOPs counter
   [https://github.com/Lyken17/pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)

### Dataset

Coral reef images and annotations are used for educational and research purposes in marine biology and AI for conservation. If you use this codebase with publicly available coral datasets, please cite the original data sources appropriately.

---

## Ethical Considerations

### Environmental Responsibility

This project explicitly addresses the environmental cost of AI through:

1. **Carbon Tracking**: All model training includes real-time CO2 emission monitoring
2. **Break-Even Analysis**: Quantifies when AI benefits outweigh computational costs
3. **Optimization Recommendations**: Provides actionable steps to reduce carbon footprint
4. **Transparency**: All emissions data is logged and made available in reports

### Data Privacy and Bias

- **Geographic Bias**: Model is trained on specific reef locations and may not generalize globally
- **Temporal Bias**: Coral bleaching patterns change over time; model should be regularly retrained
- **Species Representation**: Performance may vary across different coral species

### Limitations

1. **Not a substitute for field expertise**: Always validate AI predictions with marine biology experts
2. **Requires quality data**: Performance degrades with poor lighting or water clarity
3. **Computational cost**: Training requires significant energy; inference is lightweight
4. **False positives/negatives**: ~39% of bleaching predictions may have >10% error

---

## License

This project is intended for educational and research purposes. If you use this code or methodology, please provide appropriate attribution.

---

**Last Updated**: October 2025
**Version**: 2.0 (with carbon tracking and improved model architecture)
