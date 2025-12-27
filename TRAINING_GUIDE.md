# COVID-19 Classification Training Guide

Complete guide for training the COVID-19 classifier with RadImageNet transfer learning.

## 📋 Quick Start

### Stage 1: Train Classifier (Frozen Backbone)
```bash
# Train with default settings (10 epochs)
python train.py --stage 1

# Custom epochs
python train.py --stage 1 --epochs 8

# Custom batch size and learning rate
python train.py --stage 1 --batch-size 64 --lr 0.001
```

### Stage 2: Fine-tune Full Model (Optional)
```bash
# Fine-tune entire model (automatically loads Stage 1 weights)
python train.py --stage 2

# Custom settings
python train.py --stage 2 --epochs 15 --lr 1e-5
```

### Evaluate Model
```bash
# Evaluate on CT test set
python evaluate.py --checkpoint checkpoints/stage1/best_model.pth

# Evaluate on X-ray test set (domain gap analysis)
python evaluate.py --checkpoint checkpoints/stage1/best_model.pth --domain xray

# Save predictions
python evaluate.py --checkpoint checkpoints/stage1/best_model.pth --save-predictions
```

---

## 🔧 Setup

### 1. Get RadImageNet Weights (Recommended)

RadImageNet provides better performance than ImageNet for medical imaging.

**Option A: Download from Official Repository**
```bash
# Create weights directory
mkdir -p weights

# Download from RadImageNet repository
# Visit: https://github.com/BMEII-AI/RadImageNet
# Or use wget (if direct link available):
wget -O weights/RadImageNet-ResNet50.pth <DOWNLOAD_URL>
```

**Option B: Use ImageNet (Fallback)**
If RadImageNet weights are unavailable, the code will automatically fall back to ImageNet pretrained weights. Performance will be 3-5% lower on medical images.

### 2. Verify Data Paths
```bash
# Check that datasets are accessible
python config.py
```

Expected output:
```
✓ CT Data: ~/.cache/kagglehub/.../Chest_CT
✓ X-ray Data: ~/.cache/kagglehub/.../Processed_XRay
```

### 3. Test Model
```bash
# Test model creation and forward pass
python model.py
```

---

## 📊 Training Pipeline

### Stage 1: Classifier Training (5-10 epochs)

**What happens:**
- Backbone (ResNet-50) is **frozen** ❄️
- Only the final classification layer (2048→2 neurons) is trained
- Fast convergence (10-15 minutes on GPU)
- Uses class weights to handle slight CT imbalance

**Configuration:**
```python
Epochs: 10
Batch Size: 32
Learning Rate: 1e-3
Optimizer: Adam
Scheduler: Cosine Annealing
```

**Expected Results:**
```
Training samples: 586
Validation samples: 60
Test samples: 200

Expected accuracy: 85-95% (with ImageNet)
Expected accuracy: 90-97% (with RadImageNet)
```

### Stage 2: Full Model Fine-tuning (Optional)

**What happens:**
- Backbone is **unfrozen** 🔥
- Entire model is fine-tuned end-to-end
- Slower convergence, higher final accuracy
- Uses much lower learning rate (1e-5)

**Configuration:**
```python
Epochs: 20
Batch Size: 16 (smaller for stability)
Learning Rate: 1e-5
Optimizer: SGD with momentum
Scheduler: Step LR
```

**When to use Stage 2:**
- Stage 1 accuracy < 90%
- You have time for longer training
- You want maximum performance

---

## 📈 Monitoring Training

### Real-time Progress
Training shows progress bars with live metrics:
```
Epoch 5/10 [Train]: 100%|████████| loss: 0.2134, acc: 91.32%
Epoch 5/10 [Val]:   100%|████████| loss: 0.1876, acc: 93.33%
```

### Training History
All metrics are saved to `checkpoints/stage1/training_history.json`:
```json
{
  "train_loss": [0.5234, 0.3421, ...],
  "train_acc": [78.34, 85.67, ...],
  "val_loss": [0.4567, 0.2890, ...],
  "val_acc": [80.00, 88.33, ...],
  "learning_rate": [0.001, 0.0009, ...]
}
```

### Checkpoints
```
checkpoints/stage1/
├── best_model.pth      # Best validation accuracy
├── latest_model.pth    # Most recent epoch
└── training_history.json
```

### Early Stopping
Training stops automatically if validation loss doesn't improve for 5 consecutive epochs.

---

## 🎯 Evaluation Metrics

### Comprehensive Metrics

Running `evaluate.py` computes:

1. **Overall Metrics**
   - Accuracy
   - Precision (weighted)
   - Recall (weighted)
   - F1-Score (weighted)
   - AUC-ROC

2. **Per-Class Metrics**
   - Precision per class
   - Recall per class
   - F1-Score per class

3. **Confusion Matrix**
   - True Positives / True Negatives
   - False Positives / False Negatives

### Visualizations

Generated automatically:
```
results/eval_ct/
├── metrics.json              # All metrics
├── confusion_matrix.png      # Confusion matrix heatmap
├── roc_curve.png            # ROC curve with AUC
├── per_class_metrics.png    # Bar chart of per-class performance
└── predictions.json         # All predictions (if --save-predictions)
```

---

## 🌉 Domain Gap Analysis

### Test on Target Domain (X-rays)

After training on CT scans, evaluate on X-rays to measure domain shift:

```bash
python evaluate.py --checkpoint checkpoints/stage1/best_model.pth --domain xray
```

**Expected behavior:**
- CT test accuracy: 90-95%
- X-ray test accuracy: 60-75% (significant drop due to domain gap)

This drop is expected! It shows the domain adaptation challenge.

### Domain Adaptation (Next Steps)

To improve X-ray performance:

1. **DANN (Domain-Adversarial Neural Networks)**
   - Train domain classifier alongside task classifier
   - Encourages domain-invariant features

2. **CORAL (Correlation Alignment)**
   - Align second-order statistics between domains
   - Simple and effective

3. **Fine-tuning on Target Domain**
   - Use X-ray data for supervised fine-tuning
   - Requires labeled X-ray samples

4. **CycleGAN**
   - Translate CT → X-ray appearance
   - More complex but powerful

---

## 💡 Tips & Best Practices

### For Better Performance

1. **Use RadImageNet weights** - Significant boost over ImageNet
2. **Monitor validation closely** - Watch for overfitting
3. **Try different learning rates** - Tune for your specific data
4. **Use mixed precision training** - Faster on modern GPUs (enabled by default)
5. **Increase batch size if possible** - Better gradient estimates

### Troubleshooting

**Low training accuracy:**
- Check learning rate (try 1e-4 or 1e-2)
- Increase epochs
- Verify data augmentation isn't too aggressive

**Overfitting (train >> val accuracy):**
- Reduce learning rate
- Add dropout (modify model.py)
- Use more aggressive augmentation
- Enable label smoothing in config.py

**Out of memory:**
- Reduce batch size: `--batch-size 16`
- Disable mixed precision: set `use_amp: False` in config.py
- Use smaller images (requires dataloader changes)

**Training too slow:**
- Enable mixed precision (default)
- Increase batch size if memory allows
- Reduce num_workers if CPU bottleneck

---

## 📁 File Structure

```
CTXray/
├── model.py              # Model definition with RadImageNet loading
├── train.py              # Training script for both stages
├── evaluate.py           # Comprehensive evaluation
├── config.py             # All hyperparameters and paths
├── dataloader.py         # CT scan dataset loader
├── xray_dataloader.py    # X-ray dataset loader
├── TRAINING_GUIDE.md     # This file
├── checkpoints/          # Saved models
│   ├── stage1/
│   └── stage2/
├── logs/                 # Training logs
├── results/              # Evaluation results
│   ├── eval_ct/
│   └── eval_xray/
└── weights/              # Pretrained weights
    └── RadImageNet-ResNet50.pth
```

---

## 🚀 Complete Workflow

### Full Training Pipeline

```bash
# 1. Test model (optional)
python model.py

# 2. Stage 1: Train classifier (10 epochs, ~15 min)
python train.py --stage 1

# 3. Evaluate on CT test set
python evaluate.py --checkpoint checkpoints/stage1/best_model.pth

# 4. Evaluate domain gap on X-rays
python evaluate.py --checkpoint checkpoints/stage1/best_model.pth --domain xray

# 5. (Optional) Stage 2: Fine-tune full model
python train.py --stage 2

# 6. (Optional) Re-evaluate after fine-tuning
python evaluate.py --checkpoint checkpoints/stage2/best_model.pth
```

### Expected Timeline
- Stage 1 training: 10-15 minutes (GPU) / 1-2 hours (CPU)
- Stage 2 training: 30-45 minutes (GPU) / 3-4 hours (CPU)
- Evaluation: 1-2 minutes per domain

---

## 📊 Expected Results Summary

### With ImageNet Pretrained Weights

| Metric | CT Train | CT Val | CT Test | X-ray Test (Gap) |
|--------|----------|--------|---------|------------------|
| Accuracy | 92-96% | 88-93% | 85-92% | 60-70% |
| F1-Score | 0.92-0.96 | 0.88-0.93 | 0.85-0.92 | 0.60-0.70 |
| AUC-ROC | 0.96-0.99 | 0.92-0.97 | 0.90-0.96 | 0.70-0.80 |

### With RadImageNet Pretrained Weights

| Metric | CT Train | CT Val | CT Test | X-ray Test (Gap) |
|--------|----------|--------|---------|------------------|
| Accuracy | 95-98% | 92-96% | 90-97% | 65-75% |
| F1-Score | 0.95-0.98 | 0.92-0.96 | 0.90-0.97 | 0.65-0.75 |
| AUC-ROC | 0.98-0.99 | 0.95-0.98 | 0.93-0.98 | 0.75-0.85 |

*Note: Actual results may vary based on random initialization and data splits*

---

## 🔄 Resume Training

If training is interrupted:

```bash
# Resume from latest checkpoint
python train.py --stage 1 --resume checkpoints/stage1/latest_model.pth

# Resume with different learning rate
python train.py --stage 1 --resume checkpoints/stage1/latest_model.pth --lr 5e-4
```

---

## 📞 Next Steps

After completing Stage 1 training:

1. ✅ Evaluate on CT test set → Baseline performance
2. ✅ Evaluate on X-ray test set → Measure domain gap
3. 🚧 Implement domain adaptation (DANN/CORAL/MMD)
4. 🚧 Fine-tune on X-ray data
5. 🚧 Compare adapted model vs baseline on X-ray test set

---

## ❓ FAQ

**Q: Should I use Stage 2?**  
A: Only if Stage 1 accuracy is < 90% or you want maximum performance.

**Q: How long does training take?**  
A: Stage 1: ~15 minutes (GPU), Stage 2: ~45 minutes (GPU)

**Q: Can I train without GPU?**  
A: Yes, but it will be 10-20x slower. Use smaller batch size.

**Q: What if I don't have RadImageNet weights?**  
A: Code will automatically use ImageNet. Performance drop is ~3-5%.

**Q: How do I know if I'm overfitting?**  
A: If train accuracy >> validation accuracy (gap > 10%)

**Q: Should I use early stopping?**  
A: Yes, it's enabled by default with patience=5 epochs

---

**Ready to train? Start with:**
```bash
python train.py --stage 1
```

Good luck! 🚀
