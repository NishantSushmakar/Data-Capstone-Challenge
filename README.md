# Coral Bleaching Detection System

## Overview

This repository contains two complementary approaches for automated coral bleaching detection in underwater imagery. Each approach addresses different aspects of coral reef health monitoring and is maintained in its own directory with detailed documentation.

### Approach 1: Deep Learning (EfficientNet + Carbon Tracking)
**📁 `EfficientNet+Carbon_tracking+Comparison/`**

Production-ready semantic segmentation with environmental impact tracking.

- **Speed**: ~50ms per image (GPU)
- **Output**: Segmentation maps, bleaching ratios, coral coverage
- **Special Feature**: Real-time carbon footprint monitoring

➡️ **[Full Documentation](./EfficientNet+Carbon_tracking+Comparison/README.md)**

---

### Approach 2: Random Forest + Explainable AI
**📁 `EfficientNet+RF+Explain/`**

Interpretable pixel-level classification with LIME explanations.

- **Processing**: CPU-friendly (no GPU required)
- **Output**: Pixel classifications, feature importance, natural language explanations
- **Special Feature**: LIME + LLM interpretability

➡️ **[Full Documentation](./EfficientNet+RF+Explain/README.md)**

---

## Quick Comparison

| Feature | Deep Learning | Random Forest |
|---------|--------------|---------------|
| **Speed** | Fast (GPU) | Moderate (CPU) |
| **Output** | Segmentation + metrics | Classifications + explanations |
| **Interpretability** | Limited | High (LIME + LLM) |
| **GPU Required** | Yes (training) | No |
| **Carbon Tracking** | ✅ | ❌ |
| **Explainability** | ❌ | ✅ |


---

## Installation

```bash
# Clone repository
git clone https://github.com/YourUsername/Data-Capstone-Challenge.git
cd Data-Capstone-Challenge

# Choose your approach and follow its README:
cd EfficientNet+Carbon_tracking+Comparison/  # OR
cd EfficientNet+RF+Explain/
```

---

## Dataset Structure

Both approaches expect:

```
data/
├── images/              # RGB coral images (.jpg)
├── masks_bleached/      # Bleached coral masks (.png)
└── masks_non_bleached/  # Healthy coral masks (.png)
```

---

## Documentation

Each approach has its own detailed README.To easily understand and implement, please follow:
- **Approach 1 (Deep Learning)**: [EfficientNet+Carbon_tracking+Comparison/README.md](./EfficientNet+Carbon_tracking+Comparison/README.md)
- **Approach 2 (Random Forest)**: [EfficientNet+RF+Explain/README.md](./EfficientNet+RF+Explain/README.md)

---

**Last Updated**: October 2025 | **License**: Educational & Research Use