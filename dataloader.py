import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from typing import Tuple, Optional
import numpy as np


class CTScanDataset(Dataset):
    """
    Dataset class for CT scan images (COVID vs Non-COVID)
    """
    def __init__(self, root_dir: str, split: str = 'train', transform: Optional[transforms.Compose] = None, combine_test_val: bool = False):
        """
        Args:
            root_dir: Root directory containing the data (e.g., /home/sgram/CTXray/data/ct_scans/Chest_CT)
            split: One of 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
            combine_test_val: If True and split='val', combines val and test sets
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.combine_test_val = combine_test_val
        
        # Define class mappings
        self.classes = ['NONCOVID_CT', 'COVID_CT']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build file list
        self.samples = []
        self._load_samples()
        
        print(f"[{split.upper()}] Loaded {len(self.samples)} images")
        print(f"  - COVID: {sum([1 for _, label in self.samples if label == 1])}")
        print(f"  - Non-COVID: {sum([1 for _, label in self.samples if label == 0])}")
    
    def _load_samples(self):
        """Load all image paths and labels"""
        # Determine which splits to load
        splits_to_load = [self.split]
        if self.combine_test_val and self.split == 'val':
            splits_to_load = ['val', 'test']
        
        for split in splits_to_load:
            split_dir = os.path.join(self.root_dir, split)
            
            for class_name in self.classes:
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.exists(class_dir):
                    print(f"Warning: Directory not found: {class_dir}")
                    continue
                    
                label = self.class_to_idx[class_name]
                
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, label))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Return the number of samples per class"""
        labels = [label for _, label in self.samples]
        return {
            'NONCOVID_CT': labels.count(0),
            'COVID_CT': labels.count(1)
        }


def get_transforms(split: str = 'train', img_size: int = 224) -> transforms.Compose:
    """
    Get appropriate transforms for each split
    
    Args:
        split: One of 'train', 'val', or 'test'
        img_size: Target image size
    
    Returns:
        transforms.Compose object
    """
    if split == 'train':
        # Training: Balanced augmentation with moderate brightness variation
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            # Moderate brightness/contrast to handle some distribution shift without destroying learning
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/Test: no augmentation
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    combine_test_val: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders
    
    Args:
        data_root: Root directory containing Chest_CT folder
        batch_size: Batch size for dataloaders
        img_size: Target image size
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        combine_test_val: If True, combines test and val into single validation set
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Create datasets
    train_dataset = CTScanDataset(
        root_dir=data_root,
        split='train',
        transform=get_transforms('train', img_size),
        combine_test_val=False
    )
    
    val_dataset = CTScanDataset(
        root_dir=data_root,
        split='val',
        transform=get_transforms('val', img_size),
        combine_test_val=combine_test_val
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def get_class_weights(dataset: CTScanDataset) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance
    
    Args:
        dataset: CTScanDataset instance
    
    Returns:
        Tensor of class weights
    """
    distribution = dataset.get_class_distribution()
    total = sum(distribution.values())
    
    weights = torch.tensor([
        total / (len(distribution) * distribution['NONCOVID_CT']),
        total / (len(distribution) * distribution['COVID_CT'])
    ], dtype=torch.float32)
    
    return weights


if __name__ == "__main__":
    # Test the dataloaders
    print("=" * 60)
    print("Testing CT Scan DataLoaders")
    print("=" * 60)
    
    data_root = "/home/sgram/.cache/kagglehub/datasets/sampathlonka86/chestctscans/versions/1/Chest_CT"
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=16,
        img_size=224,
        num_workers=2
    )
    
    print("\n" + "=" * 60)
    print("DataLoader Statistics:")
    print("=" * 60)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test loading a batch
    print("\n" + "=" * 60)
    print("Testing batch loading:")
    print("=" * 60)
    
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Labels in batch: {labels.tolist()}")
    
    # Calculate class weights
    print("\n" + "=" * 60)
    print("Class Weights (for handling imbalance):")
    print("=" * 60)
    weights = get_class_weights(train_loader.dataset)
    print(f"Non-COVID weight: {weights[0]:.4f}")
    print(f"COVID weight: {weights[1]:.4f}")
    
    print("\n" + "=" * 60)
    print("DataLoaders created successfully! ✓")
    print("=" * 60)
