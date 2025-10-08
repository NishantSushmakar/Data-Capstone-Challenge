# Carbon Analysis Results

## Key Numbers

### Training Phase (One-Time Cost)

- **Emissions**: 0.12 kg CO₂eq
- **Duration**: 1.4 hours
- Equivalent to
  - Driving 0.55 km in a car
  - 15 smartphone charges
  - 0.006 trees needed for 1 year to offset

### Deployment Phase (Per Image)

- **Emissions**: 0.011 g CO₂ per image
- **Speed**: 28 ms per image (36 images/second)

### Model Size

- **Parameters**: 12.4 million
- **Model File Size**: 48 MB
- **Computational Power**: 6.1 billion FLOPs

------



## Answers to SLE Questions

### Question 1: What is the computational footprint?

**Training Cost** (happens once):

```
Emissions: 0.12 kg CO₂eq
Time: 1.4 hours
```

**Inference Cost** (happens every time):

```
Per image: 0.011 g CO₂
Per 1000 images: 11 g CO₂ = 0.011 kg
```

**Key Point**: Training cost is small (like driving 0.5 km), and inference cost is tiny (0.011 g per image).

------

### Question 2: What environmental benefits and timeline?

**Traditional Method** (assumed):

```
Boat survey: ~50 kg CO₂ per survey
Survey coverage: ~1,000 images
Cost per image: 0.05 kg = 50 g CO₂
```

**AI Method**:

```
Training (once): 0.12 kg
Per image: 0.011 g CO₂
```

**Benefits Timeline**:

- Break-even after **2-3 images** ✅
- After 1,000 images: Save **49.8 kg CO₂** (equivalent to 229 km driving)
- After 10,000 images: Save **499 kg CO₂** (equivalent to 2,296 km driving)

------

### Question 3: When do costs outweigh benefits?

**Answer**: AI becomes beneficial after just **2-3 images**.

**Break-Even Calculation**:

```
Training cost: 0.12 kg
Savings per image: 50 g - 0.011 g = 49.989 g
Break-even: 0.12 kg ÷ 0.049989 kg = 2.4 images
```

**Important Note**: This assumes traditional surveys cost 50 g CO₂ per image. You should discuss this assumption in your essay (see "Writing Tips" below).

**When AI is NOT justified**:

- Small studies (< 10 images)
- One-time assessments
- When traditional methods are already very efficient

------

### Question 4: How to reduce environmental footprint?

**Current Model Already Uses**:

- ✅ Efficient architecture (EfficientNet-B2)
- ✅ Batch processing (8 images at once)
- ✅ GPU acceleration
- ✅ Transfer learning (pre-trained weights)

**Further Optimization Possible**:

1. **Model Compression** (30-50% reduction)
   - Pruning: Remove unnecessary parameters
   - Quantization: Use INT8 instead of FLOAT32
   - Knowledge distillation: Train smaller "student" model
2. **Training Strategy** (50-70% reduction)
   - Use more transfer learning
   - Reduce training epochs (60 → 30)
   - Train during clean energy hours
3. **Deployment** (2-4× improvement)
   - Edge devices (solar-powered underwater cameras)
   - TensorRT optimization
   - Larger batch sizes for bulk processing

