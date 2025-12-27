# COVID-19 Domain Adaptation: CT → X-ray

🏆 **Complete domain adaptation pipeline** for COVID-19 classification using RadImageNet pretrained ResNet-50.

## 🎯 Project Goal

Train a COVID-19 classifier on **CT scans** (source domain), then adapt it to **X-ray images** (target domain) using domain adaptation techniques.

✅ **Status**: All methods implemented and evaluated! Achieved **97.83%** test accuracy on X-ray images.

---

## 📊 Dataset Status

### ✅ Source Domain: CT Scans
- **Total**: 846 images
- **Train**: 586 (269 COVID, 317 Non-COVID)
- **Val**: 60 (30 each)
- **Test**: 200 (100 each)
- **Location**: `~/.cache/kagglehub/.../Chest_CT`
- **Note**: Use download_data.py to download CT dataset from Kaggle.

### ✅ Target Domain: X-rays  
- **Total**: 920 images
- **Train**: 644 (322 COVID-19, 322 Normal)
- **Val**: 138 (69 each)
- **Test**: 138 (69 each)
- **Location**: `~/.cache/kagglehub/.../Processed_XRay`
- **Preprocessing**: PNEUMONIA class removed, classes balanced
- **Note**: Use download_xray_data.py to download CT dataset from Kaggle.

See DATA_README for more detail about datasets.

---

## 🏆 Domain Adaptation Results

We trained a COVID-19 classifier on CT scans and successfully adapted it to X-ray images using multiple domain adaptation techniques.

### 📊 Performance Comparison

| Method | Validation Acc | Test Acc | Improvement | Training Time | Complexity |
|--------|----------------|----------|-------------|---------------|------------|
| **Baseline** (CT→X-ray, no adaptation) | 60.87% | 60.87% | - | - | ⭐ |
| **Fine-tuning** | 97.10% | **97.83%** 🥇 | **+36.96%** | ~5 min | ⭐⭐ |
| **DANN** (Adversarial) | 95.65% | 97.10% 🥈 | +36.23% | ~5 min | ⭐⭐⭐⭐ |
| **CORAL** (Covariance) | 90.58% | 88.41% 🥉 | +27.54% | ~4 min | ⭐⭐ |
| **MMD** (Kernel-based) | 90.58% | 84.06% | +23.19% | ~5 min | ⭐⭐⭐ |

### 🎯 Key Findings

1. **Fine-tuning wins**: Simple supervised fine-tuning achieved the best performance (97.83%)
2. **DANN close second**: Domain-adversarial training nearly matched fine-tuning (97.10%) with better generalization potential
3. **Statistical methods struggle**: CORAL and MMD improved over baseline but fell short for strong domain shifts like CT→X-ray
4. **Strong domain gap**: 27-37% improvement needed to bridge CT→X-ray distribution shift

### 📂 Method Implementations

- **FineTuned/**: Simple supervised fine-tuning on X-ray data
- **DANN/**: Domain-adversarial neural network with gradient reversal layer
- **CORAL/**: Correlation alignment of feature covariance matrices
- **MMD/**: Maximum mean discrepancy with Gaussian kernels

Each folder contains:
- Training script
- Model definition
- Checkpoints
- Training logs
- Comprehensive results summary

---

## 🚀 Quick Start

### 1. Train Source Model (CT Scans)
```bash
# Train ResNet-50 on CT scans (Stage 1)
python train.py --stage 1

# Evaluate on CT test set
python evaluate.py --checkpoint checkpoints/stage1/best_model.pth
```

### 2. Run Domain Adaptation

```bash
# Fine-tuning (Best: 97.83%)
cd FineTuned && python finetune_xray.py

# DANN - Domain Adversarial (97.10%)
cd DANN && python train_dann.py

# CORAL - Correlation Alignment (88.41%)
cd CORAL && python train_coral.py

# MMD - Maximum Mean Discrepancy (84.06%)
cd MMD && python train_mmd.py
```

### 3. Evaluate Domain Gap
```bash
# Baseline: CT model on X-rays (60.87%)
python evaluate.py --checkpoint checkpoints/stage1/best_model.pth --domain xray

# Fine-tuned model (97.83%)
python evaluate.py --checkpoint FineTuned/checkpoints/best_model.pth --domain xray
```

---

## 📁 Project Structure

```
CTXray/
├── model.py                 # ResNet-50 with RadImageNet loading
├── train.py                 # Training script (Stage 1 & 2)
├── evaluate.py              # Comprehensive evaluation
├── config.py                # All hyperparameters
├── dataloader.py            # CT scan dataset
├── xray_dataloader.py       # X-ray dataset
├── process_xray_data.py     # X-ray preprocessing
├── visualize_datasets.py    # Dataset visualization
├── DATA_README.md           # Dataset documentation
├── QUICKSTART.md            # Quick start guide
├── TRAINING_GUIDE.md        # Detailed training guide
│
├── FineTuned/              # 🥇 Simple fine-tuning (97.83%)
│   ├── finetune_xray.py
│   ├── checkpoints/
│   ├── results/
│   └── RESULTS_SUMMARY.md
│
├── DANN/                   # 🥈 Domain adversarial (97.10%)
│   ├── train_dann.py
│   ├── dann_model.py
│   ├── checkpoints/
│   ├── results/
│   └── RESULTS_SUMMARY.md
│
├── CORAL/                  # 🥉 Correlation alignment (88.41%)
│   ├── train_coral.py
│   ├── coral_loss.py
│   ├── checkpoints/
│   ├── results/
│   └── RESULTS_SUMMARY.md
│
├── MMD/                    # Maximum mean discrepancy (84.06%)
│   ├── train_mmd.py
│   ├── mmd_loss.py
│   ├── checkpoints/
│   ├── results/
│   └── RESULTS_SUMMARY.md
│
├── checkpoints/            # Source model checkpoints
│   └── stage1/
│       └── best_model.pth  (88.33% on CT)
│
├── results/                # Baseline evaluations
│   ├── BASELINE_RESULTS.md
│   └── domain_comparison.png
│
└── weights/                # Pretrained weights
    └── RadImageNet-ResNet50.pth  (optional)
```

---

## 🔧 Model Architecture

### Transfer Learning Strategy

```
┌─────────────────────────────────────────┐
│   ResNet-50 (RadImageNet Pretrained)    │
│                                          │
│   ┌──────────────────────────────┐      │
│   │   Backbone (Frozen ❄️)       │      │
│   │   - conv1, bn1, relu, maxpool│      │
│   │   - layer1 (256 filters)     │      │
│   │   - layer2 (512 filters)     │      │
│   │   - layer3 (1024 filters)    │      │
│   │   - layer4 (2048 filters)    │      │
│   │   - avgpool                  │      │
│   └──────────────────────────────┘      │
│              ↓                           │
│   ┌──────────────────────────────┐      │
│   │   Classifier Head (Trainable)│      │
│   │   Linear(2048 → 2)           │      │
│   │   [Non-COVID, COVID]         │      │
│   └──────────────────────────────┘      │
└─────────────────────────────────────────┘

Total Parameters: 23,512,130
Trainable (Stage 1): 4,098 (0.02%)
```

### Training Stages

**Stage 1: Classifier Training (5-10 epochs)**
- Freeze ResNet-50 backbone
- Train only final layer (2048 → 2)
- Fast convergence, good initialization
- Expected: 90-97% accuracy on CT test set

**Stage 2: Full Fine-tuning (Optional, 15-20 epochs)**
- Unfreeze entire model
- End-to-end training with low LR (1e-5)
- Marginal improvement, longer training

---

## 📊 Results Achieved

### Source Domain (CT Scans)

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 88.33% |
| **Training Approach** | Layer4-only unfrozen (63.79% params) |
| **Convergence** | ~10 epochs |
| **Model** | ResNet-50 + RadImageNet weights |

### Target Domain (X-ray) - After Adaptation

| Method | Test Accuracy | COVID Recall | Normal Recall | F1-Score |
|--------|---------------|--------------|---------------|----------|
| **Baseline** (no adaptation) | 60.87% | 21.74% ⚠️ | 100.00% | 0.36 |
| **Fine-tuning** | **97.83%** 🏆 | 97.10% | 98.55% | 0.98 |
| **DANN** | 97.10% | ~96% | ~98% | 0.97 |
| **CORAL** | 88.41% | ~85% | ~91% | 0.88 |
| **MMD** | 84.06% | ~80% | ~88% | 0.84 |

**Critical Discovery**: Baseline model had catastrophic 78% false negative rate for COVID cases, showing conservative bias toward "Normal" predictions.

### Domain Gap Analysis

- **Initial Gap**: 88.33% (CT) → 60.87% (X-ray) = **27.46% drop**
- **Fine-tuning**: Closed gap completely (+36.96%)
- **DANN**: Nearly closed gap (+36.23%) with better generalization potential
- **CORAL/MMD**: Partial improvements but insufficient for strong domain shift

---

## 🌉 Domain Adaptation Methods

### ✅ Implemented Methods

#### 1. **Fine-tuning** (97.83%) 🥇
**Approach**: Simple supervised learning on X-ray data
- Load CT checkpoint as initialization
- Unfreeze all layers
- Train with low learning rate (1e-4)
- **Pros**: Best performance, simplest implementation
- **Cons**: May overfit to X-ray, less generalizable

#### 2. **DANN - Domain Adversarial Neural Networks** (97.10%) 🥈
**Approach**: Adversarial training with gradient reversal layer
- Feature extractor tries to fool domain classifier
- Domain classifier learns to distinguish CT vs X-ray
- Gradient reversal creates domain-invariant features
- **Pros**: Theoretically sound, generalizes to new domains
- **Cons**: Complex implementation, requires careful tuning

#### 3. **CORAL - Correlation Alignment** (88.41%) 🥉
**Approach**: Align second-order statistics (covariance)
- Match covariance matrices between CT and X-ray features
- Simple statistical alignment
- **Pros**: Easy to implement, interpretable
- **Cons**: Weak for strong domain shifts, only aligns covariance

#### 4. **MMD - Maximum Mean Discrepancy** (84.06%)
**Approach**: Kernel-based distribution matching
- Minimize distance between mean embeddings
- Uses Gaussian kernels with multiple bandwidths
- **Pros**: Theoretically elegant, multi-scale matching
- **Cons**: Weak for strong shifts, validation-test gap

### 🎓 Lessons Learned

1. **Strong domain shifts need strong methods**: Fine-tuning or DANN
2. **Statistical alignment insufficient**: CORAL/MMD only improved 23-27%
3. **Adversarial training works**: DANN nearly matched fine-tuning
4. **Simplicity wins**: Fine-tuning achieved best results with least complexity

---

## 💡 Getting RadImageNet Weights

RadImageNet provides **3-5% better accuracy** than ImageNet on medical images.

### Option 1: Official Repository
```bash
# Visit GitHub repository
# https://github.com/BMEII-AI/RadImageNet

# Download ResNet-50 weights
# Save to: /home/sgram/CTXray/weights/RadImageNet-ResNet50.pth
```

### Option 2: Use ImageNet (Automatic Fallback)
If RadImageNet is unavailable, code automatically uses ImageNet:
- No code changes needed
- Slightly lower performance (~3-5%)
- Still gets 85-92% accuracy on CT test set

---

## 📚 Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training walkthrough
- **[DATA_README.md](DATA_README.md)** - Dataset details and statistics
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start commands

---

## 🔍 Model Testing

### Test Model Creation
```bash
python model.py
```

### Test Configuration
```bash
python config.py
```

### Test Dataloaders
```bash
# CT scans
python dataloader.py

# X-rays
python xray_dataloader.py
```

### Visualize Datasets
```bash
python visualize_datasets.py
```

---

## 📈 Training Pipeline

### Complete Workflow

```bash
# 1. Verify setup
python config.py

# 2. Stage 1: Train classifier (10 epochs, ~15 min on GPU)
python train.py --stage 1

# 3. Evaluate on CT test set (source domain)
python evaluate.py --checkpoint checkpoints/stage1/best_model.pth

# 4. Measure domain gap on X-rays (target domain)
python evaluate.py --checkpoint checkpoints/stage1/best_model.pth --domain xray

# 5. (Optional) Stage 2: Fine-tune full model
python train.py --stage 2

# 6. Implement domain adaptation (next phase)
# ... DANN / CORAL / MMD ...
```

---

## 🎯 Project Status

### ✅ Completed

**Data & Infrastructure**
- [x] Downloaded and preprocessed CT scan dataset (846 images)
- [x] Downloaded and preprocessed X-ray dataset (920 images)
- [x] Created balanced datasets (removed PNEUMONIA class)
- [x] Implemented dataloaders with augmentation
- [x] Created visualizations and statistics

**Source Model Training**
- [x] Trained ResNet-50 on CT scans (88.33% validation)
- [x] Tested layer configurations (layer4-only optimal)
- [x] Comprehensive evaluation and metrics

**Domain Adaptation**
- [x] Established baseline (60.87% - significant gap)
- [x] Implemented Fine-tuning → 97.83% ✅
- [x] Implemented DANN → 97.10% ✅
- [x] Implemented CORAL → 88.41% ✅
- [x] Implemented MMD → 84.06% ✅
- [x] Comprehensive comparison and analysis

**Documentation**
- [x] Method explanations and intuitions
- [x] Training logs and checkpoints
- [x] Results summaries for all methods
- [x] Complete README

### 🚀 Future Work

1. **External Validation**: Test on different hospitals/datasets
2. **Ensemble Methods**: Combine Fine-tuning + DANN predictions
3. **Explainability**: Add Grad-CAM visualizations
4. **Clinical Deployment**: Uncertainty quantification, PACS integration
5. **Additional Methods**: Self-training, meta-learning, transformer-based approaches

---

## 🛠️ Dependencies

Already installed:
- PyTorch
- torchvision
- NumPy
- Matplotlib
- PIL
- kagglehub
- tqdm

Additional for evaluation:
- scikit-learn (for metrics)
- seaborn (for confusion matrix)

```bash
pip install scikit-learn seaborn
```

---

## 📊 Monitoring

### During Training
- Real-time progress bars with loss/accuracy
- Validation metrics every epoch
- Automatic checkpointing (best + latest)
- Early stopping (patience=5)
- Learning rate scheduling

### After Training
- Training history JSON (all epochs)
- Best model checkpoint
- Confusion matrices
- ROC curves
- Per-class metrics
- Prediction files

---

## ⚡ Performance Tips

1. **Use CUDA if available** - 10-20x faster than CPU
2. **Enable mixed precision** - Enabled by default, faster on modern GPUs
3. **Adjust batch size** - Increase if you have GPU memory
4. **Use RadImageNet** - Significant boost over ImageNet
5. **Monitor early** - Check first few epochs, adjust LR if needed

---

## 🤝 Contributing

This is a research project. To extend:

1. **Add new domain adaptation methods**
   - Implement in new file (e.g., `dann.py`)
   - Follow similar structure to `train.py`
   - Add configuration to `config.py`

2. **Experiment with architectures**
   - Try DenseNet-121, EfficientNet, Vision Transformers
   - Modify `model.py` to support new architectures

3. **Improve preprocessing**
   - Experiment with augmentation strategies
   - Test different normalization schemes

---

## 📞 Support

For issues or questions:
1. Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed help
2. Verify data paths with `python config.py`
3. Test components individually (model.py, dataloader.py)

---

## 📝 Citation

If using this code or approach, please cite:

- **RadImageNet**: Mei et al., "RadImageNet: An Open Radiologic Deep Learning Research Dataset"
- **Domain Adaptation**: Ganin et al., "Domain-Adversarial Training of Neural Networks" (if using DANN)

---

## ✨ Quick Commands Summary

```bash
# ============================================
# Stage 1: Train source model on CT scans
# ============================================
python train.py --stage 1
python evaluate.py --checkpoint checkpoints/stage1/best_model.pth

# ============================================
# Stage 2: Measure domain gap
# ============================================
python evaluate.py --checkpoint checkpoints/stage1/best_model.pth --domain xray
# Expected: ~60% accuracy (27% gap)

# ============================================
# Stage 3: Domain Adaptation (choose one)
# ============================================

# Option 1: Fine-tuning (Best - 97.83%)
cd FineTuned && python finetune_xray.py

# Option 2: DANN (Strong - 97.10%)
cd DANN && python train_dann.py

# Option 3: CORAL (Moderate - 88.41%)
cd CORAL && python train_coral.py

# Option 4: MMD (Moderate - 84.06%)
cd MMD && python train_mmd.py
```

---

## 📈 Results Summary

**Starting Point**: CT model → 60.87% on X-rays ❌  
**After Adaptation**: Fine-tuning → 97.83% on X-rays ✅  
**Improvement**: +36.96% (COVID recall: 21.74% → 97.10%)

🏆 **Recommended**: Use **Fine-tuning** for best performance or **DANN** for better generalization to unseen domains.

---

**Project Complete!** 🎉 All domain adaptation methods implemented and evaluated.
