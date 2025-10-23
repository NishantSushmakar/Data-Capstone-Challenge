# Coral Bleaching Detection System

## Overview

Coral bleaching is a critical indicator of reef health, but traditional monitoring through manual surveys is time-consuming and limited in scale. This project provides a **two-step machine learning pipeline** for automated coral bleaching detection from underwater imagery, enabling rapid assessment of reef health across large areas.

**Key Features:**
- Automated coral segmentation and pixel-level classification
- Interpretable predictions with LIME explanations
- Carbon footprint monitoring
- Scalable from single images to large-scale reef surveys

---

## 🔄 Two-Step Pipeline

### Step 1: Semantic Segmentation with Carbon Tracking
**📁 `EfficientNet+Carbon_tracking+Comparison/`**

Production-ready semantic segmentation with environmental impact tracking.

- **Speed**: ~50ms per image (GPU)
- **Output**: Segmentation maps, bleaching ratios, coral coverage
- **Special Feature**: Real-time carbon footprint monitoring

➡️ **[Full Documentation](./EfficientNet+Carbon_tracking+Comparison/README.md)**

---

### Step 2: Random Forest + Explainable AI
**📁 `EfficientNet+RF+Explain/`**

Interpretable pixel-level classification with LIME explanations.

- **Processing**: CPU-friendly (no GPU required)
- **Output**: Pixel classifications, feature importance, natural language explanations
- **Special Feature**: LIME + LLM interpretability

➡️ **[Full Documentation](./EfficientNet+RF+Explain/README.md)**

---

## Installation

```bash
# Clone repository
git clone https://github.com/YourUsername/Data-Capstone-Challenge.git
cd Data-Capstone-Challenge

# If you want to create segmentation maps and track carbon footprint
cd EfficientNet+Carbon_tracking+Comparison/  

# If you want to classify using Random Forest with explainability
cd EfficientNet+RF+Explain/
```

---

## Dataset Structure

```
data/
├── images/              # RGB coral images (.jpg)
├── masks_bleached/      # Bleached coral masks (.png)
└── masks_non_bleached/  # Healthy coral masks (.png)
```

---

## Documentation

Each approach has its own detailed README.To easily understand and implement, please follow:
- **Step 1 (Deep Learning)**: [EfficientNet+Carbon_tracking+Comparison/README.md](./EfficientNet+Carbon_tracking+Comparison/README.md)
- **Step 2 (Random Forest)**: [EfficientNet+RF+Explain/README.md](./EfficientNet+RF+Explain/README.md)

---

**Last Updated**: October 2025 | **License**: Educational & Research Use