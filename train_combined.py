"""
Training with combined train+val sets and stronger augmentation for higher accuracy.
Uses 5-fold cross-validation for better model selection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold

from model import create_model
from dataloader import CTScanDataset
from config import CT_DATA_PATH

# Strong augmentation for better generalization
def get_strong_transforms(split='train', img_size=224):
    """Enhanced data augmentation"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def train_with_combined_data():
    """Train on train+val combined with cross-validation"""
    
    # Load combined train+val data
    train_dataset = CTScanDataset(str(CT_DATA_PATH), 'train', get_strong_transforms('train'))
    val_dataset = CTScanDataset(str(CT_DATA_PATH), 'val', get_strong_transforms('val'))
    
    # Combine datasets
    combined_samples = train_dataset.samples + val_dataset.samples
    labels = [label for _, label in combined_samples]
    
    print(f"Combined dataset: {len(combined_samples)} images")
    print(f"  COVID: {sum(labels)}")
    print(f"  Non-COVID: {len(labels) - sum(labels)}")
    
    # 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_models = []
    fold_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(combined_samples, labels)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold+1}/5")
        print(f"{'='*70}")
        
        # Create fold datasets
        train_samples = [combined_samples[i] for i in train_idx]
        val_samples = [combined_samples[i] for i in val_idx]
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.stack([get_strong_transforms('train')(
                    __import__('PIL.Image', fromlist=['Image']).open(path).convert('RGB')
                ) for path, _ in train_samples]),
                torch.tensor([label for _, label in train_samples])
            ),
            batch_size=32, shuffle=True, num_workers=4
        )
        
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.stack([get_strong_transforms('val')(
                    __import__('PIL.Image', fromlist=['Image']).open(path).convert('RGB')
                ) for path, _ in val_samples]),
                torch.tensor([label for _, label in val_samples])
            ),
            batch_size=32, shuffle=False, num_workers=4
        )
        
        # Train model for this fold
        model = create_model(
            pretrained_path='weights/RadImageNet-ResNet50.pth',
            num_classes=2,
            freeze_backbone=False,  # Full fine-tuning
            device='cuda'
        )
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
        scaler = GradScaler()
        
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        # Train for this fold
        for epoch in range(30):
            # Training
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/30'):
                images, labels = images.cuda(), labels.cuda()
                
                optimizer.zero_grad()
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
            
            train_acc = 100. * correct / total
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.cuda(), labels.cuda()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, pred = outputs.max(1)
                    total += labels.size(0)
                    correct += pred.eq(labels).sum().item()
            
            val_acc = 100. * correct / total
            
            print(f"Epoch {epoch+1}: Train {train_acc:.2f}% | Val {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model for this fold
                torch.save(model.state_dict(), f'checkpoints/fold{fold+1}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            scheduler.step()
        
        print(f"\nFold {fold+1} Best Val Acc: {best_val_acc:.2f}%")
        fold_accs.append(best_val_acc)
    
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Fold accuracies: {[f'{acc:.2f}%' for acc in fold_accs]}")
    print(f"Mean: {np.mean(fold_accs):.2f}% ± {np.std(fold_accs):.2f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    train_with_combined_data()
