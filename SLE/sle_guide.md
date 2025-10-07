# 🌍 Carbon Tracking for Coral Bleaching Detection - SLE Guide

## Quick Start

### 1. Installation
```bash
# Install carbon tracking libraries
pip install codecarbon thop

# Verify installation
python -c "import codecarbon; print('✓ CodeCarbon installed')"
python -c "import thop; print('✓ THOP installed')"
```

### 2. File Structure
```
your_project/
├── improved_model.py              # Your existing model
├── carbon_tracking_integration.py # New: Carbon tracking tools
├── train_with_carbon_tracking.py  # New: Enhanced training script
└── carbon_reports/                # Auto-generated reports
    ├── emissions.csv
    ├── coral_bleaching_carbon_report.json
    └── sle_environmental_impact_report.json
```

### 3. Run Training with Carbon Tracking
```bash
# Simply run the new training script
python train_with_carbon_tracking.py
```

---

## What You Get for Your SLE Essay

### 📊 Quantitative Data Generated

#### **Question 1: Computational Resource Footprint**

You'll receive exact measurements:
- **Training emissions**: X.XX kg CO₂eq
- **Training duration**: X.XX hours  
- **Model complexity**: X.XX million parameters, X.XX GFLOPs
- **Inference cost**: X.XXXXXX g CO₂ per prediction
- **Inference latency**: X.XX ms per image

**Context comparisons** (automatically calculated):
- Equivalent to driving X.X km in a car
- X.XX trees needed for 1 year to offset
- X,XXX smartphone charges

#### **Question 2: Environmental Benefits Timeline**

The code calculates:
- **Break-even point**: After analyzing X,XXX images
- **Payback period**: After X.X traditional surveys
- **Long-term savings**: X.XX kg CO₂ saved per year

#### **Question 3: When Costs Outweigh Benefits**

Automated analysis provides:
- **Cost-benefit curves**: Visual comparison of AI vs traditional methods
- **Critical thresholds**: When training emissions become unjustified
- **Scenario analysis**: Different deployment scales

#### **Question 4: Footprint Reduction Practices**

Evidence-based recommendations:
- **Model optimization**: Potential 30-50% parameter reduction
- **Training efficiency**: 50-70% reduction via transfer learning
- **Inference optimization**: 2-4x speedup through quantization
- **Energy source impact**: Renewable vs fossil fuel grid comparison

---

## Understanding the Output Files

### 1. `coral_bleaching_carbon_report.json`
Raw carbon tracking data:
```json
{
  "training": [{
    "emissions_kg_co2eq": 0.1234,
    "duration_hours": 2.5
  }],
  "inference": [{
    "emissions_per_prediction_g": 0.000123,
    "time_per_prediction_ms": 15.2
  }]
}
```

### 2. `sle_environmental_impact_report.json`
Structured for your essay:
```json
{
  "question_1_computational_footprint": {
    "training_phase": {...},
    "deployment_phase": {...}
  },
  "question_3_cost_benefit_analysis": {
    "break_even_images": 5000,
    "verdict": "Breaks even after..."
  },
  "question_4_optimization_recommendations": [...]
}
```

### 3. `sle_environmental_analysis.png`
Four-panel visualization:
- Training emissions in context
- Break-even analysis graph
- Deployment efficiency metrics
- Lifecycle carbon footprint

---

## Key Metrics Explained

### Carbon Emissions
- **kg CO₂eq**: Kilograms of CO₂ equivalent (includes all greenhouse gases)
- **Scope**: Only direct computational emissions (Scope 2)
- **Exclusions**: Manufacturing of hardware, data center infrastructure

### Model Efficiency
- **FLOPs**: Floating Point Operations (computational complexity)
- **Parameters**: Number of trainable weights
- **Latency**: Time to process one image
- **Throughput**: Images processed per second

### Break-Even Analysis
```
Total Cost = Training Cost (one-time) + Inference Cost × Number of Images
Break-even when: AI Total Cost < Traditional Total Cost
```

---

## Customizing for Your Context

### Location-Specific Carbon Intensity

**Netherlands** (default):
```python
carbon_tracker = CarbonAwareTrainer(
    country_iso_code="NLD",  # Netherlands
    region="North Holland"
)
```

**Other locations**: Change ISO code
- USA: "USA", region="California"
- UK: "GBR", region="England"  
- Australia: "AUS", region="New South Wales"

### Traditional Method Assumptions

Adjust based on your research:
```python
traditional_assumptions = {
    'emissions_kg_per_survey': 50.0,    # Boat + equipment
    'images_per_survey': 1000,           # Coverage per survey
    'survey_frequency_per_year': 12      # Monitoring frequency
}
```

**Sources to cite**:
- Boat fuel consumption: X liters/hour × Y kg CO₂/liter
- Survey duration: Z hours per survey
- Equipment transport: Consider flights, vehicles

---

## Writing Your SLE Essay with the Data

### Structure Template

#### **Question 1: Resource Footprint**

"Training the EfficientNet-B2 coral bleaching model consumed **[X.XX kg CO₂eq]** over **[X.X hours]**, equivalent to **[driving X km]** or **[Y smartphone charges]**. The model's **[X.X million parameters]** and **[Y.Y GFLOPs]** indicate moderate computational demand compared to state-of-the-art vision models."

**Add nuance**:
- Compare to GPT-3 (training: ~500,000 kg CO₂) to show relative efficiency
- Note that EfficientNet is specifically designed for low-resource scenarios
- Discuss one-time vs recurring costs

#### **Question 3: Ethical Justification**

"The break-even point occurs after **[X,XXX images]**, equivalent to **[Y traditional surveys]**. Given typical monitoring frequencies of **[Z surveys/year]**, the AI system becomes carbon-neutral within **[W months]**. Beyond this point, each additional year of deployment saves approximately **[V kg CO₂]**."

**Critical analysis**:
- What if the model needs retraining? (concept drift, new coral species)
- Deployment scale matters: Large reef systems justify training costs
- Small-scale studies may not reach break-even

#### **Question 4: Reduction Practices**

"Model optimization techniques could reduce the footprint:
1. **Transfer learning**: Reduce training emissions by **[50-70%]** by fine-tuning pretrained models
2. **Quantization**: Decrease inference energy by **[2-4×]** through INT8 precision
3. **Edge deployment**: Eliminate cloud transmission costs (solar-powered underwater cameras)
4. **Efficient architectures**: MobileNet variants consume **[30-50% less]** energy than EfficientNet"

---

## Comparing to Literature

### Benchmark Your Results

**Typical ML Training Emissions**:
- Small CNN (yours): ~0.1-1 kg CO₂  ✓ Your result
- BERT training: ~280 kg CO₂
- GPT-3 training: ~500,000 kg CO₂

**Inference Costs**:
- Your model: ~0.0001 g CO₂/image
- Cloud API (GPT-4V): ~0.01-0.1 g CO₂/query (estimate)

### Citations to Use

1. **Strubell et al. (2019)**: "Energy and Policy Considerations for Deep Learning in NLP"
2. **Schwartz et al. (2020)**: "Green AI"
3. **Patterson et al. (2021)**: "Carbon Emissions and Large Neural Network Training"
4. **Lacoste et al. (2019)**: "Quantifying the Carbon Emissions of Machine Learning" (CodeCarbon creators)

---

## Troubleshooting

### Issue: "codecarbon not tracking"
**Solution**: Check if GPU usage is detected:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Issue: "Emissions seem too low"
**Reason**: Short training time, efficient model
**Verification**: Compare to manual calculation:
```
GPU Power (W) × Training Hours × Grid Carbon Intensity (kg CO₂/kWh)
Example: 160W × 2.5h × 0.5 kg/kWh = 0.2 kg CO₂
```

### Issue: "Break-even point is infinite"
**Meaning**: AI inference costs more per image than traditional methods
**Action**: 
1. Verify traditional method assumptions
2. Consider batch processing to amortize costs
3. Discuss in essay as a limitation

---

## Advanced: Carbon-Optimal Training

### 1. Choose Low-Carbon Training Times
```python
# Train when grid is cleanest (more renewable energy)
# For Netherlands: typically midday (solar peak)
import datetime
current_hour = datetime.datetime.now().hour
if 11 <= current_hour <= 15:
    print("✓ Training during clean energy peak")
```

### 2. Early Stopping
```python
# Stop training when validation improvement plateaus
# Saves unnecessary epochs
patience = 5  # Stop if no improvement for 5 epochs
```

### 3. Model Selection
```python
# Compare architectures by efficiency score
efficiency_score = FLOPs × Latency × Model_Size / Accuracy²
# Lower is better
```

---

## Example Essay Excerpt

> "To quantify the environmental costs of our automated coral bleaching detection system, we employed CodeCarbon to track carbon emissions throughout the development lifecycle. Training the EfficientNet-B2 model consumed 0.142 kg CO₂eq over 2.3 hours on an NVIDIA RTX 2060 GPU in the Netherlands (grid carbon intensity: 0.39 kg CO₂/kWh). This is equivalent to driving 0.65 kilometers in a gasoline vehicle or approximately 18 smartphone charges (Lacoste et al., 2019).
>
> In the deployment phase, inference costs 0.000087 g CO₂ per image with a latency of 15.2 ms, enabling processing of 65 images per second. Comparing this to traditional boat-based surveys, which emit approximately 50 kg CO₂ per survey analyzing 1,000 images (0.05 g per image), our AI system breaks even after processing 2,840 images—less than three traditional surveys.
>
> However, this analysis assumes continuous deployment at scale. For small-scale research projects analyzing fewer than 3,000 images, the training emissions may never be offset, raising ethical questions about when AI deployment is environmentally justified..."

---

## Final Checklist for Your SLE Essay

- [ ] Actual emissions data from your model (not estimates)
- [ ] Comparison to equivalent activities (km driven, trees, etc.)
- [ ] Break-even analysis with realistic assumptions
- [ ] Discussion of one-time vs recurring costs
- [ ] Model efficiency metrics (FLOPs, parameters, latency)
- [ ] Optimization recommendations with quantified benefits
- [ ] Critical analysis of when AI is NOT justified
- [ ] Citations to carbon tracking methodology (CodeCarbon paper)
- [ ] Visualizations showing cost-benefit trade-offs
- [ ] Limitations of your analysis (scope boundaries, assumptions)

---

## Questions for Your SLE Discussion

Use these to demonstrate critical thinking:

1. **Should training emissions be amortized over model lifetime or counted upfront?**
2. **How does model retraining frequency affect the cost-benefit calculation?**
3. **Are there equity concerns? (Resource-rich institutions can afford training, resource-poor cannot)**
4. **What if edge deployment enables new monitoring that wasn't feasible before?**
5. **How do we value prevented coral loss vs carbon emissions?**

---

## Additional Resources

- **CodeCarbon Documentation**: https://codecarbon.io/
- **ML Carbon Calculator**: https://mlco2.github.io/impact/
- **Green Algorithms**: http://www.green-algorithms.org/
- **Papers with Code**: https://paperswithcode.com/ (compare model efficiency)

---

**Good luck with your SLE essay! 🌊🪸**

The quantitative evidence you now have will make your ethical arguments much more compelling and defensible.