# CORAL (CORrelation ALignment) - Results Summary

**Method**: CORAL (Correlation Alignment)  
**Paper**: "Return of Frustratingly Easy Domain Adaptation" (Sun et al., 2016)  
**Date**: December 27, 2025

---

## 1. What is CORAL?

CORAL aligns the **second-order statistics (covariance matrices)** between source and target domain features.

### Key Idea:
- Different domains have different feature distributions
- Match the **covariance** (how features correlate with each other) between domains
- Forces features to have similar statistical properties

### CORAL Loss:
```
CORAL_loss = || C_source - C_target ||²_F / (4 * d²)
```

Where:
- `C_source`: Covariance matrix of source features (CT)
- `C_target`: Covariance matrix of target features (X-ray)
- `||.||_F`: Frobenius norm (sum of squared differences)
- `d`: Feature dimension (2048)

### Training:
```
Total_loss = Classification_loss + λ * CORAL_loss
```

- **Classification loss**: Predict COVID/Normal correctly
- **CORAL loss**: Align covariance matrices
- **λ** (lambda): Weight for CORAL loss (1.0)

---

## 2. Performance Results

### Comparison Table:

| Method | Baseline (CT→X-ray) | Fine-tuning | DANN | **CORAL** |
|--------|---------------------|-------------|------|----------|
| **Validation Accuracy** | 60.87% | 97.10% | 95.65% | **90.58%** |
| **Test Accuracy** | 60.87% | 97.83% | 97.10% | **88.41%** |
| **Improvement** | - | +36.96% | +36.23% | **+27.54%** |

### CORAL Performance:
- **Best epoch**: 4 (Val: 90.58%)
- **Test accuracy**: 88.41%
- **Training accuracy**: 96.35% (final epoch)
- **Early stopping**: Triggered at epoch 19 (patience 15)

### Key Observations:
1. ✅ **Significant improvement** over baseline (+27.54%)
2. ❌ **Lower than fine-tuning** (-9.42% vs 97.83%)
3. ❌ **Lower than DANN** (-8.69% vs 97.10%)
4. ⚠️ **Converged early** (epoch 4) but continued training

---

## 3. Training Metrics

### Loss Progression:

| Epoch | Train Class Loss | Train CORAL Loss | Val Loss | Val Acc | Train Acc |
|-------|-----------------|------------------|----------|---------|-----------|
| 1 | 0.6786 | 0.0001 | 0.4008 | 89.13% | 86.98% |
| **4** | **0.3406** | **0.0001** | **0.2978** | **90.58%** ✅ | **93.58%** |
| 7 | 0.2803 | 0.0001 | 0.3297 | 89.86% | 95.14% |
| 16 | 0.2263 | 0.0001 | 0.2916 | 90.58% | 96.35% |
| 19 | 0.2406 | 0.0001 | 0.2884 | 90.58% | 95.23% |

### CORAL Loss Analysis:
- **Very small values**: ~0.0001 throughout training
- **Stable**: No significant change across epochs
- **Question**: Is CORAL loss too weak? (Weight = 1.0)

### Training Behavior:
- **Fast convergence**: Best model at epoch 4
- **Stable plateau**: Val acc stayed ~90% after epoch 4
- **No overfitting**: Train acc 96% vs Val acc 91% (reasonable gap)
- **Early stopping**: Patience exhausted at epoch 19

---

## 4. Why CORAL Underperformed

### Hypothesis 1: CORAL Loss Too Weak
- CORAL loss values: ~0.0001
- Classification loss: ~0.3-0.7
- **CORAL contribution**: Only 0.03% of total loss
- **Solution**: Increase λ (e.g., 10.0 or 100.0)

### Hypothesis 2: Covariance Alignment Not Sufficient
- CORAL only aligns **second-order** statistics (covariance)
- Doesn't align **first-order** statistics (mean)
- Doesn't enforce **domain invariance** like DANN
- **Result**: Features still domain-specific

### Hypothesis 3: Limited Data
- CT: 586 samples, X-ray: 644 samples
- Small batches (32 samples)
- **Covariance estimation**: Requires more samples for accurate matrix
- **Batch covariance**: May be noisy/unstable

### Hypothesis 4: Feature Distribution Mismatch
- CT vs X-ray: Very different imaging modalities
- **Strong domain shift**: Covariance alone can't bridge gap
- **DANN advantage**: Adversarial learning forces invariance
- **Fine-tuning advantage**: Directly learns X-ray patterns

---

## 5. Comparison: Fine-tuning vs DANN vs CORAL

### Fine-tuning (97.83%):
✅ **Simplest**: Just train on target data  
✅ **Best performance**: Highest test accuracy  
✅ **Fast convergence**: Epoch 6  
❌ **Overfits to X-ray**: May not generalize to new domains  

### DANN (97.10%):
✅ **Domain-invariant**: Features work across domains  
✅ **Excellent performance**: Nearly matches fine-tuning  
✅ **Theoretical guarantee**: Adversarial game  
❌ **Complex**: GRL, domain classifier, careful tuning  

### CORAL (88.41%):
✅ **Simple**: Just add CORAL loss  
✅ **Fast**: No adversarial training  
✅ **Interpretable**: Clear statistical alignment  
❌ **Lower performance**: 9% worse than fine-tuning  
❌ **Weak loss**: CORAL contribution very small  

---

## 6. Architecture Details

### Model Structure:
```
CoralModel:
  ├── feature_extractor (ResNet-50 backbone)
  │   ├── Conv1, BN1, ReLU, MaxPool
  │   ├── Layer1 (3 blocks)
  │   ├── Layer2 (4 blocks)
  │   ├── Layer3 (6 blocks)
  │   ├── Layer4 (3 blocks)
  │   └── AdaptiveAvgPool2d
  │
  └── classifier (Final FC layer)
      └── Linear(2048 → 2)

CORAL Loss:
  ├── Compute CT feature covariance: C_ct
  ├── Compute X-ray feature covariance: C_xray
  └── Loss = || C_ct - C_xray ||²_F / (4 * d²)
```

### Parameters:
- **Total**: 23,512,130 (23.5M)
- **Trainable**: 23,512,130 (100%)
- **Feature dim**: 2048

### Training Configuration:
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealing (T_max=20, eta_min=1e-6)
- **Batch size**: 32
- **Epochs**: 20 (early stopped at 19)
- **CORAL weight**: 1.0
- **Mixed precision**: Enabled (AMP)
- **Device**: cuda:1

---

## 7. When to Use CORAL?

### ✅ Use CORAL when:
1. **Simple baseline needed**: Quick domain adaptation experiment
2. **Limited computational resources**: No adversarial training
3. **Interpretability important**: Clear statistical alignment goal
4. **Weak domain shift**: Small distribution differences

### ❌ Avoid CORAL when:
1. **Strong domain shift**: CT↔X-ray has large differences
2. **Best performance required**: Fine-tuning or DANN better
3. **Small batch sizes**: Covariance estimation unstable
4. **Multiple domains**: DANN more flexible

### 💡 Improvements to Try:
1. **Increase CORAL weight**: λ = 10.0 or 100.0
2. **Add mean alignment**: Match first + second order statistics
3. **Larger batches**: 64 or 128 for stable covariance
4. **Deep CORAL**: Apply at multiple layers
5. **Combine with DANN**: CORAL + adversarial training

---

## 8. Files Generated

### Checkpoints:
- `checkpoints/best_model.pth` (Epoch 4, Val 90.58%)

### Results:
- `results/coral_training.log` (Full training log)
- `results/training_history.json` (Metrics per epoch)
- `results/summary.json` (Final performance summary)
- `RESULTS_SUMMARY.md` (This file)

### Code:
- `coral_loss.py` (CORAL loss implementation)
- `train_coral.py` (Training script)

---

## 9. Conclusion

### Summary:
- ✅ **CORAL improved** over baseline: 60.87% → 88.41% (+27.54%)
- ❌ **Underperformed** vs fine-tuning: 88.41% vs 97.83% (-9.42%)
- ❌ **Underperformed** vs DANN: 88.41% vs 97.10% (-8.69%)

### Why CORAL Failed:
1. **CORAL loss too weak**: Only 0.0001 vs classification loss 0.3-0.7
2. **Strong domain shift**: CT↔X-ray too different for covariance alignment alone
3. **Limited expressiveness**: Only aligns second-order statistics
4. **Small batch size**: Unstable covariance estimation with 32 samples

### Recommendations:
1. **For this project**: Use **fine-tuning** (97.83%) or **DANN** (97.10%)
2. **For CORAL improvement**: Increase λ to 10-100, try larger batches
3. **For future work**: Combine CORAL + DANN for hybrid approach

### Final Ranking:
1. 🥇 **Fine-tuning**: 97.83% (simplest + best)
2. 🥈 **DANN**: 97.10% (domain-invariant)
3. 🥉 **CORAL**: 88.41% (simple but weak)
4. 📉 **Baseline**: 60.87% (no adaptation)

---

**Next Steps**: Try increasing CORAL weight (λ=10 or λ=100) to see if stronger alignment helps, or consider combining CORAL with adversarial training for hybrid approach.
