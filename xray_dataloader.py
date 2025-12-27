"""
Dataloader for X-ray dataset (COVID vs Normal)
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from typing import Tuple, Optional
import numpy as np


class XRayDataset(Dataset):
    """
    Dataset class for X-ray images (COVID vs Normal)
    """
    def __init__(self, root_dir: str, split: str = 'train', transform: Optional[transforms.Compose] = None):
        """
        Args:
            root_dir: Root directory containing the data
            split: One of 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Define class mappings
        self.classes = ['NORMAL', 'COVID19']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build file list
        self.samples = []
        self._load_samples()
        
        print(f"[{split.upper()}] Loaded {len(self.samples)} X-ray images")
        print(f"  - COVID-19: {sum([1 for _, label in self.samples if label == 1])}")
        print(f"  - Normal: {sum([1 for _, label in self.samples if label == 0])}")
    
    def _load_samples(self):
        """Load all image paths and labels"""
        split_dir = os.path.join(self.root_dir, self.split)
        
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
            'NORMAL': labels.count(0),
            'COVID19': labels.count(1)
        }


def get_xray_transforms(split: str = 'train', img_size: int = 224) -> transforms.Compose:
    """
    Get appropriate transforms for X-ray images
    
    Args:
        split: One of 'train', 'val', or 'test'
        img_size: Target image size
    
    Returns:
        transforms.Compose object
    """
    if split == 'train':
        # Training: augmentation for better generalization
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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


def create_xray_dataloaders(
    data_root: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for X-ray data
    
    Args:
        data_root: Root directory containing train/val/test folders
        batch_size: Batch size for dataloaders
        img_size: Target image size
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = XRayDataset(
        root_dir=data_root,
        split='train',
        transform=get_xray_transforms('train', img_size)
    )
    
    val_dataset = XRayDataset(
        root_dir=data_root,
        split='val',
        transform=get_xray_transforms('val', img_size)
    )
    
    test_dataset = XRayDataset(
        root_dir=data_root,
        split='test',
        transform=get_xray_transforms('test', img_size)
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def get_xray_class_weights(dataset: XRayDataset) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance
    
    Args:
        dataset: XRayDataset instance
    
    Returns:
        Tensor of class weights
    """
    distribution = dataset.get_class_distribution()
    total = sum(distribution.values())
    
    weights = torch.tensor([
        total / (len(distribution) * distribution['NORMAL']),
        total / (len(distribution) * distribution['COVID19'])
    ], dtype=torch.float32)
    
    return weights


if __name__ == "__main__":
    # Test the dataloaders
    print("=" * 70)
    print("Testing X-Ray DataLoaders")
    print("=" * 70)
    
    data_root = "/home/sgram/.cache/kagglehub/datasets/prashant268/chest-xray-covid19-pneumonia/versions/2/Processed_XRay"
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_xray_dataloaders(
        data_root=data_root,
        batch_size=16,
        img_size=224,
        num_workers=2
    )
    
    print("\n" + "=" * 70)
    print("DataLoader Statistics:")
    print("=" * 70)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test loading a batch
    print("\n" + "=" * 70)
    print("Testing batch loading:")
    print("=" * 70)
    
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Labels in batch: {labels.tolist()}")
    
    # Calculate class weights
    print("\n" + "=" * 70)
    print("Class Weights (for handling imbalance):")
    print("=" * 70)
    weights = get_xray_class_weights(train_loader.dataset)
    print(f"Normal weight: {weights[0]:.4f}")
    print(f"COVID-19 weight: {weights[1]:.4f}")
    
    print("\n" + "=" * 70)
    print("X-Ray DataLoaders created successfully! ✓")
    print("=" * 70)
