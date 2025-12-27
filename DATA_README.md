# COVID-19 Domain Adaptation Dataset - Setup Complete

## Overview

This project contains two datasets for domain adaptation from CT scans to X-rays for COVID-19 classification:
- **Source Domain**: CT Scans (COVID vs Non-COVID)
- **Target Domain**: X-rays (COVID vs Normal)

---

# CT Scan Dataset (Source Domain)

## Dataset Information

**Source**: [Kaggle - Chest CT Scans](https://www.kaggle.com/datasets/sampathlonka86/chestctscans)

**Location**: Data is stored in `/home/sgram/.cache/kagglehub/datasets/sampathlonka86/chestctscans/versions/1/Chest_CT`

### Dataset Statistics

| Split | COVID-19 | Non-COVID | Total |
|-------|----------|-----------|-------|
| Train | 269      | 317       | 586   |
| Val   | 30       | 30        | 60    |
| Test  | 100      | 100       | 200   |
| **Total** | **399** | **447** | **846** |

### Class Distribution

- **Training set**: Slightly imbalanced (54.1% Non-COVID, 45.9% COVID)
- **Validation set**: Perfectly balanced (50% each)
- **Test set**: Perfectly balanced (50% each)

### Class Weights (for handling imbalance)
- Non-COVID weight: 0.9243
- COVID weight: 1.0892

## Data Structure

```
/home/sgram/.cache/kagglehub/datasets/sampathlonka86/chestctscans/versions/1/Chest_CT/
├── train/
│   ├── COVID_CT/       (269 images)
│   └── NONCOVID_CT/    (317 images)
├── val/
│   ├── COVID_CT/       (30 images)
│   └── NONCOVID_CT/    (30 images)
└── test/
    ├── COVID_CT/       (100 images)
    └── NONCOVID_CT/    (100 images)
```

## Usage

### Basic Usage

```python
from dataloader import create_dataloaders

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_root="/home/sgram/.cache/kagglehub/datasets/sampathlonka86/chestctscans/versions/1/Chest_CT",
    batch_size=32,
    img_size=224,
    num_workers=4
)

# Iterate through batches
for images, labels in train_loader:
    # images: torch.Tensor of shape [batch_size, 3, 224, 224]
    # labels: torch.Tensor of shape [batch_size], values: 0 (Non-COVID), 1 (COVID)
    pass
```

### Custom Transforms

```python
from dataloader import CTScanDataset, get_transforms
from torch.utils.data import DataLoader

# Create dataset with custom transforms
train_dataset = CTScanDataset(
    root_dir="/home/sgram/.cache/kagglehub/datasets/sampathlonka86/chestctscans/versions/1/Chest_CT",
    split='train',
    transform=get_transforms('train', img_size=256)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)
```

### Get Class Weights

```python
from dataloader import create_dataloaders, get_class_weights

train_loader, _, _ = create_dataloaders(
    data_root="/home/sgram/.cache/kagglehub/datasets/sampathlonka86/chestctscans/versions/1/Chest_CT"
)

# For use with weighted loss functions
class_weights = get_class_weights(train_loader.dataset)
# class_weights: tensor([0.9243, 1.0892])
```

## Data Preprocessing

### Training Set Augmentation
- Resize to 224×224
- Random horizontal flip (p=0.5)
- Random rotation (±10 degrees)
- Color jitter (brightness & contrast ±20%)
- Random affine translation (±10%)
- Normalization: ImageNet statistics

### Validation/Test Set
- Resize to 224×224
- Normalization: ImageNet statistics
- No augmentation

### Normalization Values
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

## Files

- `dataloader.py` - Main dataloader implementation
- `download_data.py` - Script to download dataset from Kaggle
- `test_dataloader.py` - Visualization and testing script
- `results/sample_batch.png` - Visualization of sample batch

## Testing

To verify the dataloaders are working correctly:

```bash
cd /home/sgram/CTXray
python dataloader.py
```

To visualize samples:

```bash
python test_dataloader.py
```

## Next Steps

1. **Build Source Model** - Train baseline classifier on CT scans
2. **Get X-ray Dataset** - Download chest X-ray COVID-19 dataset
3. **Domain Adaptation** - Implement adaptation techniques (DANN, CORAL, etc.)
4. **Evaluation** - Compare performance across domains

## Notes

- Images are loaded as RGB (3 channels)
- All images are resized to 224×224 by default (configurable)
- Dataset uses ImageNet normalization for transfer learning compatibility
- Class imbalance in training set is handled via class weights
- Data lives in kagglehub cache directory (no project duplication)

---

# X-Ray Dataset (Target Domain)

## Dataset Information

**Source**: [Kaggle - Chest X-Ray COVID19 Pneumonia](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)

**Location**: `/home/sgram/.cache/kagglehub/datasets/prashant268/chest-xray-covid19-pneumonia/versions/2/Processed_XRay`

**Processing Steps**:
1. ✓ Removed PNEUMONIA class (not needed for binary classification)
2. ✓ Downsampled NORMAL class to match COVID-19 count (460 images each)
3. ✓ Created balanced train/val/test splits (70/15/15)

### Dataset Statistics

| Split | COVID-19 | Normal | Total |
|-------|----------|--------|-------|
| Train | 322      | 322    | 644   |
| Val   | 69       | 69     | 138   |
| Test  | 69       | 69     | 138   |
| **Total** | **460** | **460** | **920** |

### Class Distribution

- **All splits**: Perfectly balanced (50% COVID-19, 50% Normal)
- **Class weights**: Both 1.0 (perfectly balanced)

## Data Structure

```
/home/sgram/.cache/kagglehub/.../Processed_XRay/
├── train/
│   ├── COVID19/    (322 images)
│   └── NORMAL/     (322 images)
├── val/
│   ├── COVID19/    (69 images)
│   └── NORMAL/     (69 images)
└── test/
    ├── COVID19/    (69 images)
    └── NORMAL/     (69 images)
```

## Usage

### Basic Usage

```python
from xray_dataloader import create_xray_dataloaders

# Create X-ray dataloaders
train_loader, val_loader, test_loader = create_xray_dataloaders(
    data_root="/home/sgram/.cache/kagglehub/datasets/prashant268/chest-xray-covid19-pneumonia/versions/2/Processed_XRay",
    batch_size=32,
    img_size=224,
    num_workers=4
)

# Iterate through batches
for images, labels in train_loader:
    # images: torch.Tensor of shape [batch_size, 3, 224, 224]
    # labels: torch.Tensor of shape [batch_size], values: 0 (Normal), 1 (COVID-19)
    pass
```

### Custom Transforms

```python
from xray_dataloader import XRayDataset, get_xray_transforms
from torch.utils.data import DataLoader

# Create dataset with custom transforms
train_dataset = XRayDataset(
    root_dir="/home/sgram/.cache/kagglehub/datasets/prashant268/chest-xray-covid19-pneumonia/versions/2/Processed_XRay",
    split='train',
    transform=get_xray_transforms('train', img_size=256)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)
```

## Data Preprocessing

Same as CT scans:
- Training: Augmentation (flip, rotation, color jitter, affine)
- Val/Test: Only resize and normalize
- ImageNet normalization statistics

## Files

- `xray_dataloader.py` - X-ray dataloader implementation
- `download_xray_data.py` - Script to download dataset from Kaggle
- `process_xray_data.py` - Script to process and balance the dataset

## Testing

To verify X-ray dataloaders:

```bash
cd /home/sgram/CTXray
python xray_dataloader.py
```

---

# Domain Adaptation Pipeline

## Dataset Comparison

| Aspect | CT Scans (Source) | X-Rays (Target) |
|--------|-------------------|-----------------|
| **Total Images** | 846 | 920 |
| **Train Size** | 586 | 644 |
| **Val Size** | 60 | 138 |
| **Test Size** | 200 | 138 |
| **Balance** | Slightly imbalanced | Perfectly balanced |
| **COVID Samples** | 399 | 460 |
| **Normal/Non-COVID** | 447 | 460 |
| **Modality** | 3D volumetric | 2D projection |
| **Resolution** | High (Hounsfield) | Lower (standard X-ray) |

## Next Steps for Domain Adaptation

1. **Train Source Model**
   - Train CNN on CT scans
   - Achieve strong baseline performance
   - Save pretrained weights

2. **Evaluate Domain Gap**
   - Test CT model directly on X-rays
   - Measure performance drop
   - Analyze failure modes

3. **Implement Adaptation**
   - Choose method: DANN, CORAL, MMD, CycleGAN, etc.
   - Train with combined CT (labeled) + X-ray (unlabeled/semi-labeled)
   - Fine-tune on target domain

4. **Evaluate & Compare**
   - Source-only performance
   - Adapted model performance
   - Improvement metrics

## Project Structure

```
CTXray/
├── dataloader.py              # CT scan dataloader
├── xray_dataloader.py         # X-ray dataloader
├── download_data.py           # Download CT data
├── download_xray_data.py      # Download X-ray data
├── process_xray_data.py       # Process X-ray dataset
├── test_dataloader.py         # Visualization scripts
├── DATA_README.md             # This file
├── requirements.txt           # Dependencies
├── checkpoints/               # Model checkpoints
├── models/                    # Model architectures
└── results/                   # Results and visualizations
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- torch
- torchvision
- PIL
- numpy
- kagglehub
- matplotlib

---

**Setup Complete!** ✓ Both datasets are ready for domain adaptation experiments.
