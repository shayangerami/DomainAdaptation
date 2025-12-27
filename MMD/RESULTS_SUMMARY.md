# MMD (Maximum Mean Discrepancy) - Results Summary

**Method**: MMD (Maximum Mean Discrepancy)  
**Paper**: "Learning Transferable Features with Deep Adaptation Networks" (Long et al., 2015)  
**Date**: December 27, 2025

---

## 1. What is MMD?

MMD measures the **distance between two distributions** using kernel embeddings in a high-dimensional space.

### Key Idea:
- Map features to a high-dimensional space using kernels (Gaussian RBF)
- Compute mean of source features and mean of target features
- Minimize the distance between these means
- Forces distributions to overlap

### MMD Formula:
```
MMD²(X_s, X_t) = E[k(x_s, x_s)] - 2*E[k(x_s, x_t)] + E[k(x_t, x_t)]
```

Where:
- `k`: Gaussian kernel (measures similarity)
- `x_s`: Source features (CT)
- `x_t`: Target features (X-ray)
- `E`: Expected value (mean)

### Multi-Scale Kernels:
Uses **5 different bandwidths** to capture patterns at different scales:
- Small bandwidth: Local similarities
- Large bandwidth: Global similarities

### Training:
```
Total_loss = Classification_loss + λ * MMD_loss
```

- **Classification loss**: Predict COVID/Normal correctly
- **MMD loss**: Minimize distribution distance
- **λ** (lambda): Weight for MMD loss (1.0)

---

## 2. Performance Results

### Comparison Table:

| Method | Baseline | Fine-tuning | DANN | CORAL | **MMD** |
|--------|----------|-------------|------|-------|---------|
| **Validation Accuracy** | 60.87% | 97.10% | 95.65% | 90.58% | **90.58%** |
| **Test Accuracy** | 60.87% | 97.83% | 97.10% | 88.41% | **84.06%** |
| **Improvement** | - | +36.96% | +36.23% | +27.54% | **+23.19%** |

### MMD Performance:
- **Best epoch**: 11 (Val: 90.58%)
- **Test accuracy**: 84.06%
- **Training accuracy**: 95.23% (final epoch)
- **Epochs trained**: 20 (full)

### Key Observations:
1. ✅ **Improved over baseline** (+23.19%)
2. ❌ **Lower than fine-tuning** (-13.77% vs 97.83%)
3. ❌ **Lower than DANN** (-13.04% vs 97.10%)
4. ✅ **Similar validation to CORAL** (90.58% vs 90.58%)
5. ❌ **Worse test than CORAL** (84.06% vs 88.41%)

---

## 3. Training Metrics

### Loss Progression:

| Epoch | Train Class Loss | Train MMD Loss | Val Loss | Val Acc | Train Acc |
|-------|-----------------|----------------|----------|---------|-----------|
| 1 | 0.7643 | 0.3800 | 0.5875 | 82.61% | 84.38% |
| 5 | 0.4298 | 0.1857 | 0.4259 | 87.68% | 91.15% |
| 8 | 0.3091 | 0.1890 | 0.3472 | 89.86% | 94.53% |
| **11** | **0.2798** | **0.1650** | **0.2887** | **90.58%** ✅ | **94.53%** |
| 20 | 0.3127 | 0.1635 | 0.3499 | 89.13% | 93.92% |

### MMD Loss Analysis:
- **Starting value**: 0.3800
- **Final value**: 0.1635
- **Reduction**: 57% decrease (effective alignment)
- **Comparison to CORAL**: ~1000x larger (0.16 vs 0.0001)

### Training Behavior:
- **Convergence**: Best model at epoch 11
- **Fluctuation**: Val acc varied 87-90% after epoch 5
- **Overfitting**: Train 95% vs Val 90% (5% gap)
- **Test drop**: Val 90.58% → Test 84.06% (-6.52%)

---

## 4. Why MMD Underperformed

### Hypothesis 1: Generalization Gap
- **Validation**: 90.58%
- **Test**: 84.06%
- **Drop**: -6.52%
- **Issue**: Overfitting to validation set or unstable performance

### Hypothesis 2: Kernel Choice
- Uses Gaussian (RBF) kernel
- **Fixed bandwidths**: May not capture CT↔X-ray differences optimally
- **Alternative**: Could try learned kernels or different kernel families

### Hypothesis 3: MMD Loss Scale
- MMD loss: ~0.16-0.38
- Classification loss: ~0.27-0.76
- **Contribution**: ~30-50% of total loss
- **Question**: Is this the right balance?

### Hypothesis 4: Strong Domain Shift
- CT vs X-ray: Fundamentally different imaging
- **Mean matching**: Not sufficient for extreme shifts
- **DANN advantage**: Adversarial forces stronger invariance
- **Fine-tuning advantage**: Directly learns target patterns

---

## 5. Comparison: All Methods

### Rankings by Test Accuracy:
1. 🥇 **Fine-tuning**: 97.83% (best + simplest)
2. 🥈 **DANN**: 97.10% (domain-invariant)
3. 🥉 **CORAL**: 88.41% (covariance alignment)
4. 📉 **MMD**: 84.06% (kernel-based)
5. ❌ **Baseline**: 60.87% (no adaptation)

### Fine-tuning (97.83%):
✅ Best performance  
✅ Simplest implementation  
✅ Fast convergence (epoch 6)  
❌ May overfit to X-ray  

### DANN (97.10%):
✅ Domain-invariant features  
✅ Excellent performance  
✅ Theoretical guarantees  
❌ Complex (GRL, adversarial)  

### CORAL (88.41%):
✅ Simple (just covariance)  
✅ Fast training  
❌ Weak loss contribution  
❌ Only second-order stats  

### MMD (84.06%):
✅ Kernel-based (flexible)  
✅ Multiple scales (5 bandwidths)  
✅ Stronger than CORAL loss  
❌ Validation-test gap  
❌ Still weak vs fine-tuning  

---

## 6. Architecture Details

### Model Structure:
```
MMDModel:
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

MMD Loss:
  ├── Gaussian kernel with 5 bandwidths
  ├── Compute K_ss (source-source similarity)
  ├── Compute K_tt (target-target similarity)
  ├── Compute K_st (source-target similarity)
  └── Loss = mean(K_ss) - 2*mean(K_st) + mean(K_tt)
```

### Parameters:
- **Total**: 23,512,130 (23.5M)
- **Trainable**: 23,512,130 (100%)
- **Feature dim**: 2048

### Training Configuration:
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealing (T_max=20, eta_min=1e-6)
- **Batch size**: 32
- **Epochs**: 20 (full run)
- **MMD weight**: 1.0
- **Kernel**: Gaussian RBF (5 scales)
- **Mixed precision**: Enabled (AMP)
- **Device**: cuda:1

---

## 7. When to Use MMD?

### ✅ Use MMD when:
1. **Kernel methods preferred**: Flexible similarity measures
2. **Multi-scale matching needed**: Different levels of abstraction
3. **Theoretical guarantee**: Wants proven distribution matching
4. **Moderate domain shift**: Not too extreme

### ❌ Avoid MMD when:
1. **Strong domain shift**: CT↔X-ray too different
2. **Best performance required**: Fine-tuning or DANN better
3. **Limited data**: Small batches make mean estimation noisy
4. **Computational constraints**: Kernel computation expensive

### 💡 Improvements to Try:
1. **Increase MMD weight**: λ = 5.0 or 10.0
2. **More bandwidths**: 10+ scales instead of 5
3. **Different kernels**: Polynomial, Laplacian, learned
4. **Multi-layer MMD**: Apply at multiple network layers
5. **Combine with adversarial**: MMD + DANN hybrid

---

## 8. MMD vs CORAL Comparison

Both are **statistical alignment** methods but measure different things:

| Aspect | CORAL | MMD |
|--------|-------|-----|
| **Measures** | Covariance (second-order) | Mean embeddings (kernel space) |
| **Order** | Second-order statistics | All moments (via kernel) |
| **Loss Scale** | Very small (0.0001) | Moderate (0.16-0.38) |
| **Val Acc** | 90.58% | 90.58% (tied) |
| **Test Acc** | 88.41% ✅ | 84.06% ❌ |
| **Computation** | Fast (matrix ops) | Slower (kernel matrix) |
| **Flexibility** | Fixed (covariance) | Flexible (kernel choice) |

### Why CORAL Outperformed MMD on Test:
- More stable (simpler objective)
- Better generalization (88.41% vs 84.06%)
- MMD had larger val-test gap (6.52% vs 2.17%)

---

## 9. Files Generated

### Checkpoints:
- `checkpoints/best_model.pth` (Epoch 11, Val 90.58%)

### Results:
- `results/mmd_training.log` (Full training log)
- `results/training_history.json` (Metrics per epoch)
- `results/summary.json` (Final performance summary)
- `RESULTS_SUMMARY.md` (This file)

### Code:
- `mmd_loss.py` (MMD loss with Gaussian kernel)
- `train_mmd.py` (Training script)

---

## 10. Conclusion

### Summary:
- ✅ **MMD improved** over baseline: 60.87% → 84.06% (+23.19%)
- ❌ **Underperformed** vs fine-tuning: 84.06% vs 97.83% (-13.77%)
- ❌ **Underperformed** vs DANN: 84.06% vs 97.10% (-13.04%)
- ❌ **Underperformed** vs CORAL: 84.06% vs 88.41% (-4.35%)

### Why MMD Failed:
1. **Validation-test gap**: 90.58% → 84.06% (unstable)
2. **Strong domain shift**: CT↔X-ray too different for kernel matching
3. **Mean matching insufficient**: Needs more than distribution means
4. **Kernel limitations**: Gaussian RBF may not capture CT/X-ray differences

### Statistical Methods Struggle:
Both CORAL and MMD performed similarly (~84-88%) because:
- Only align **statistics** (covariance, means)
- Don't enforce **invariance** like DANN
- Don't **directly learn** target patterns like fine-tuning
- CT↔X-ray shift too strong for statistical tricks

### Final Ranking (All Methods):
1. 🥇 **Fine-tuning**: 97.83% (direct learning)
2. 🥈 **DANN**: 97.10% (adversarial invariance)
3. 🥉 **CORAL**: 88.41% (covariance alignment)
4. 📉 **MMD**: 84.06% (kernel matching)
5. ❌ **Baseline**: 60.87% (no adaptation)

### Insights:
- **Strong shifts need strong methods**: Fine-tuning or DANN
- **Statistical alignment weak**: CORAL/MMD ~10% worse
- **Kernel methods don't help**: MMD worse than CORAL despite flexibility
- **Validation ≠ Test**: MMD had largest generalization gap

---

**Recommendation**: For CT→X-ray adaptation, use **Fine-tuning** (simplest + best) or **DANN** (domain-invariant). Statistical methods (CORAL/MMD) insufficient for this strong domain shift.
