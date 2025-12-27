# Domain Adaptation Setup - Quick Start Guide

## ✓ Setup Complete!

Both datasets have been downloaded, processed, and are ready for domain adaptation experiments.

---

## 📊 Dataset Summary

### Source Domain: CT Scans
- **Path**: `/home/sgram/.cache/kagglehub/datasets/sampathlonka86/chestctscans/versions/1/Chest_CT`
- **Total**: 846 images (399 COVID, 447 Non-COVID)
- **Splits**: Train (586), Val (60), Test (200)
- **Dataloader**: `dataloader.py`

### Target Domain: X-Rays
- **Path**: `/home/sgram/.cache/kagglehub/datasets/prashant268/chest-xray-covid19-pneumonia/versions/2/Processed_XRay`
- **Total**: 920 images (460 COVID-19, 460 Normal)
- **Splits**: Train (644), Val (138), Test (138)
- **Dataloader**: `xray_dataloader.py`
- **Processing**: PNEUMONIA removed, NORMAL downsampled to match COVID-19

---

## 🚀 Quick Start

### Test Dataloaders

```bash
# Test CT dataloader
python dataloader.py

# Test X-ray dataloader
python xray_dataloader.py

# Visualize both datasets
python visualize_datasets.py
```

### Use in Your Code

```python
# Import dataloaders
from dataloader import create_dataloaders
from xray_dataloader import create_xray_dataloaders

# Load CT data (source domain)
ct_train, ct_val, ct_test = create_dataloaders(
    data_root="/home/sgram/.cache/kagglehub/datasets/sampathlonka86/chestctscans/versions/1/Chest_CT",
    batch_size=32,
    img_size=224
)

# Load X-ray data (target domain)
xray_train, xray_val, xray_test = create_xray_dataloaders(
    data_root="/home/sgram/.cache/kagglehub/datasets/prashant268/chest-xray-covid19-pneumonia/versions/2/Processed_XRay",
    batch_size=32,
    img_size=224
)

# Training loop
for ct_imgs, ct_labels in ct_train:
    # Train on source domain (CT scans)
    pass

for xray_imgs, xray_labels in xray_train:
    # Adapt to target domain (X-rays)
    pass
```

---

## 📁 Project Files

### Data Processing
- ✓ `download_data.py` - Download CT scans
- ✓ `download_xray_data.py` - Download X-rays
- ✓ `process_xray_data.py` - Process and balance X-ray data

### Data Loading
- ✓ `dataloader.py` - CT scan dataloader
- ✓ `xray_dataloader.py` - X-ray dataloader
- ✓ `test_dataloader.py` - Test CT dataloader
- ✓ `visualize_datasets.py` - Compare both datasets

### Documentation
- ✓ `DATA_README.md` - Comprehensive dataset documentation
- ✓ `QUICKSTART.md` - This file

### Results
- ✓ `results/sample_batch.png` - CT sample batch
- ✓ `results/domain_comparison.png` - CT vs X-ray comparison
- ✓ `results/setup_summary.png` - Setup summary statistics

---

## 📋 Label Mappings

### CT Scans
- Label 0: Non-COVID
- Label 1: COVID

### X-Rays
- Label 0: Normal
- Label 1: COVID-19

**Note**: Both use binary classification with similar semantic meaning.

---

## 🎯 Next Steps for Domain Adaptation

### 1. Train Source Model
```bash
# Create train_source.py to train on CT scans
# Save best model checkpoint
```

### 2. Evaluate Domain Gap
```bash
# Test CT model on X-rays without adaptation
# Measure performance drop
```

### 3. Implement Adaptation Method

Choose one or combine:

**A. Domain Adversarial Neural Networks (DANN)**
- Add domain classifier
- Gradient reversal layer
- Confuse source/target features

**B. Correlation Alignment (CORAL)**
- Match feature covariance
- Simple and effective
- No adversarial training

**C. Maximum Mean Discrepancy (MMD)**
- Minimize distribution distance
- Kernel-based approach

**D. CycleGAN (Pixel-level)**
- Translate X-rays to look like CTs
- Or vice versa
- Use generated images for training

### 4. Fine-tune and Evaluate
```bash
# Train adaptation model
# Evaluate on X-ray test set
# Compare: source-only vs adapted
```

---

## 🔬 Evaluation Metrics

Track these metrics:
- Accuracy
- Sensitivity (Recall for COVID class)
- Specificity
- F1-Score
- AUC-ROC
- Confusion Matrix

For domain adaptation specifically:
- **Source-only accuracy** on target domain (baseline)
- **Adapted model accuracy** on target domain
- **Improvement** = Adapted - Source-only
- **Source domain accuracy** (shouldn't degrade much)

---

## ⚙️ Configuration

All datasets use:
- Image size: 224×224
- Normalization: ImageNet stats
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- Augmentation (train only):
  - Random horizontal flip
  - Random rotation (±10°)
  - Color jitter
  - Random affine

---

## 🐛 Troubleshooting

### Out of Memory?
Reduce batch size:
```python
train_loader = create_dataloaders(..., batch_size=16)
```

### Data Loading Slow?
Adjust workers:
```python
train_loader = create_dataloaders(..., num_workers=0)  # Single thread
```

### Can't Find Data?
Check paths match your system:
```bash
ls /home/sgram/.cache/kagglehub/datasets/
```

---

## 📚 References

Datasets:
- [Chest CT Scans - Kaggle](https://www.kaggle.com/datasets/sampathlonka86/chestctscans)
- [Chest X-Ray COVID19 - Kaggle](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)

Domain Adaptation Methods:
- DANN: Ganin et al., 2016
- CORAL: Sun & Saenko, 2016
- CycleGAN: Zhu et al., 2017
- MMD: Long et al., 2015

---

**Ready to start domain adaptation!** 🚀

For detailed information, see [DATA_README.md](DATA_README.md)
