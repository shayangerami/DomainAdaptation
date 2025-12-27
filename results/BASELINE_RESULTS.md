# Domain Adaptation Baseline Results

## Model Configuration
- **Architecture**: ResNet-50 with RadImageNet pretrained weights
- **Training Strategy**: Layer4-only unfrozen (63.79% trainable params)
- **Training Data**: CT scans (586 samples, 60 validation)
- **Optimization**: Adam (lr=1e-3), Cosine annealing, Mixed precision (AMP)

---

## Source Domain Performance (CT Scans)

### Validation Set Results
- **Dataset**: 60 CT validation samples (30 COVID, 30 Non-COVID)
- **Accuracy**: **88.33%** ✅
- **Training Accuracy**: 91.67%
- **Train-Val Gap**: 3.34% (healthy generalization)

### Model Checkpoint
- Best model: `checkpoints/stage1/best_model.pth`
- Epoch: 14/20
- Validation Loss: 0.3931

---

## Target Domain Performance (X-ray Images) - BASELINE

### Test Set Results
- **Dataset**: 138 X-ray test samples (69 COVID-19, 69 Normal)
- **Accuracy**: **60.87%** ⚠️
- **Precision**: 0.7805 (weighted)
- **Recall**: 0.6087 (weighted)
- **F1-Score**: 0.5379 (weighted)
- **AUC-ROC**: 0.6387

### Per-Class Performance

#### Normal Class
- Precision: **0.5610**
- Recall: **1.0000** ✅ (detects all Normal cases)
- F1-Score: 0.7188

#### COVID-19 Class  
- Precision: **1.0000** ✅ (no false positives)
- Recall: **0.2174** ❌ (misses 78% of COVID cases)
- F1-Score: 0.3571

### Confusion Matrix
```
                Predicted
                Normal    COVID-19
Actual  Normal    69         0
        COVID-19  54        15
```

### Critical Issues
1. **Severe COVID-19 Underdetection**: Only 15/69 COVID cases detected (21.74%)
2. **54 False Negatives**: COVID cases misclassified as Normal (dangerous in medical context)
3. **Conservative Bias**: Model defaults to predicting Normal for most X-rays
4. **Poor Discrimination**: AUC-ROC 0.6387 barely exceeds random chance (0.5)

---

## Domain Gap Analysis

### Performance Drop
- **CT Validation**: 88.33%
- **X-ray Test**: 60.87%
- **Domain Gap**: **27.46%** ⚠️⚠️⚠️

### Root Causes
1. **Imaging Modality Difference**:
   - CT: 3D volumetric scans with Hounsfield units
   - X-ray: 2D projection radiographs
   - Different intensity distributions, contrast, and anatomical visualization

2. **Feature Distribution Shift**:
   - CT trained features don't transfer well to X-ray appearance
   - Ground glass opacities (CT) vs consolidation patterns (X-ray)
   - Different visual signatures for COVID-19

3. **Model Behavior**:
   - Learned to be conservative on unfamiliar X-ray patterns
   - Defaults to "Normal" when uncertain
   - High-confidence COVID predictions (precision=1.0) but very selective (recall=0.22)

---

## Next Steps: Domain Adaptation

To bridge the 27.46% performance gap, we will implement:

### 1. **Fine-tuning on X-ray Data**
- Supervised training on X-ray dataset
- Expected improvement: +15-20%

### 2. **Domain Adversarial Neural Network (DANN)**
- Learn domain-invariant features
- Gradient reversal layer for domain confusion
- Expected improvement: +10-15%

### 3. **Correlation Alignment (CORAL)**
- Align second-order statistics between domains
- Minimize covariance difference
- Expected improvement: +5-10%

### 4. **Maximum Mean Discrepancy (MMD)**
- Reduce distribution difference in feature space
- Kernel-based domain matching
- Expected improvement: +5-10%

### Target Performance
- **Goal**: 80%+ accuracy on X-ray test set
- **Gap to Close**: 20%+ improvement from baseline 60.87%

---

## Files Generated
- Metrics: `results/eval_xray/metrics.json`
- Confusion Matrix: `results/eval_xray/confusion_matrix.png`
- ROC Curve: `results/eval_xray/roc_curve.png`
- Per-class Metrics: `results/eval_xray/per_class_metrics.png`
- Evaluation Log: `results/baseline_xray_evaluation.log`

---

**Baseline Established**: December 27, 2025
**Status**: Ready for domain adaptation experiments
