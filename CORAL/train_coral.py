"""
Training script for CORAL (Correlation Alignment)

Trains on both CT (source) and X-ray (target) data:
- Both domains have labels (supervised)
- Minimizes classification loss on both domains
- Minimizes CORAL loss to align feature covariance matrices
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from coral_loss import CoralModel, coral_loss
from dataloader import create_dataloaders as create_ct_dataloaders
from xray_dataloader import create_xray_dataloaders


class CORALTrainer:
    """Trainer for CORAL domain adaptation"""
    
    def __init__(self,
                 ct_checkpoint_path,
                 ct_data_root,
                 xray_data_root,
                 output_dir='.',
                 learning_rate=1e-4,
                 num_epochs=20,
                 batch_size=32,
                 coral_weight=1.0,
                 device='cuda:1'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.results_dir = self.output_dir / 'results'
        
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n{'='*70}")
        print("CORAL (CORrelation ALignment)")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"CORAL weight: {coral_weight}")
        
        # Create CORAL model
        self.model = CoralModel(
            ct_checkpoint_path=ct_checkpoint_path,
            num_classes=2,
            device=self.device
        )
        self.model = self.model.to(self.device)
        
        # Load data
        print(f"\nLoading CT data from: {ct_data_root}")
        self.ct_train_loader, self.ct_val_loader = create_ct_dataloaders(
            data_root=ct_data_root,
            batch_size=batch_size,
            num_workers=4,
            combine_test_val=False
        )
        
        print(f"\nLoading X-ray data from: {xray_data_root}")
        self.xray_train_loader, self.xray_val_loader, self.xray_test_loader = create_xray_dataloaders(
            data_root=xray_data_root,
            batch_size=batch_size,
            num_workers=4
        )
        
        # Training setup
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.coral_weight = coral_weight
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.get_parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Early stopping
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0
        
        # History
        self.history = {
            'train_class_loss': [], 'train_coral_loss': [], 'train_total_loss': [],
            'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        
        print(f"\n✓ Optimizer: Adam (lr={learning_rate})")
        print(f"✓ Scheduler: CosineAnnealing (T_max={num_epochs})")
        print(f"✓ CT train samples: {len(self.ct_train_loader.dataset)}")
        print(f"✓ CT val samples: {len(self.ct_val_loader.dataset)}")
        print(f"✓ X-ray train samples: {len(self.xray_train_loader.dataset)}")
        print(f"✓ X-ray val samples: {len(self.xray_val_loader.dataset)}")
        print(f"✓ X-ray test samples: {len(self.xray_test_loader.dataset)}")
        print(f"{'='*70}\n")
    
    def train_epoch(self, epoch):
        """Train for one epoch with CORAL loss"""
        self.model.train()
        
        # Track metrics
        total_class_loss = 0.0
        total_coral_loss = 0.0
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        # Create iterator for X-ray data
        xray_iter = iter(self.xray_train_loader)
        
        pbar = tqdm(self.ct_train_loader, desc=f'Epoch {epoch}')
        for ct_images, ct_labels in pbar:
            # Get CT batch (source domain)
            ct_images = ct_images.to(self.device)
            ct_labels = ct_labels.to(self.device)
            batch_size = ct_images.size(0)
            
            # Get X-ray batch (target domain)
            try:
                xray_images, xray_labels = next(xray_iter)
            except StopIteration:
                xray_iter = iter(self.xray_train_loader)
                xray_images, xray_labels = next(xray_iter)
            
            xray_images = xray_images[:batch_size].to(self.device)
            xray_labels = xray_labels[:batch_size].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                # Forward pass for CT (source)
                ct_features, ct_output = self.model(ct_images)
                
                # Forward pass for X-ray (target)
                xray_features, xray_output = self.model(xray_images)
                
                # Classification losses
                ct_class_loss = self.classification_criterion(ct_output, ct_labels)
                xray_class_loss = self.classification_criterion(xray_output, xray_labels)
                class_loss = ct_class_loss + xray_class_loss
                
                # CORAL loss (align covariance matrices)
                coral_loss_value = coral_loss(ct_features, xray_features)
                
                # Total loss
                loss = class_loss + self.coral_weight * coral_loss_value
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Track metrics (combined CT + X-ray)
            combined_output = torch.cat([ct_output, xray_output], dim=0)
            combined_labels = torch.cat([ct_labels, xray_labels], dim=0)
            
            total_class_loss += class_loss.item() * (batch_size * 2)
            total_coral_loss += coral_loss_value.item() * (batch_size * 2)
            total_loss += loss.item() * (batch_size * 2)
            
            _, preds = combined_output.max(1)
            correct += preds.eq(combined_labels).sum().item()
            total_samples += combined_labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'C_loss': f'{class_loss.item():.4f}',
                'CORAL': f'{coral_loss_value.item():.4f}',
                'acc': f'{100.*correct/total_samples:.2f}%'
            })
        
        # Compute epoch metrics
        epoch_class_loss = total_class_loss / total_samples
        epoch_coral_loss = total_coral_loss / total_samples
        epoch_total_loss = total_loss / total_samples
        epoch_acc = 100.0 * correct / total_samples
        
        return epoch_class_loss, epoch_coral_loss, epoch_total_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self):
        """Validate on X-ray validation set (target domain)"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.xray_val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            _, class_output = self.model(images)
            loss = self.classification_criterion(class_output, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = class_output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / total
        val_acc = 100.0 * correct / total
        
        return val_loss, val_acc
    
    @torch.no_grad()
    def test(self):
        """Test on X-ray test set"""
        self.model.eval()
        correct = 0
        total = 0
        
        for images, labels in self.xray_test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            _, class_output = self.model(images)
            _, predicted = class_output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        test_acc = 100.0 * correct / total
        return test_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, path)
            print(f"    ✓ Saved best model to {path}")
    
    def train(self):
        """Main training loop"""
        print("Starting CORAL training...")
        print(f"{'='*70}\n")
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"Epoch {epoch}/{self.num_epochs}")
            print("-" * 70)
            
            # Train
            class_loss, coral_loss_val, total_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_class_loss'].append(class_loss)
            self.history['train_coral_loss'].append(coral_loss_val)
            self.history['train_total_loss'].append(total_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Print metrics
            print(f"  Train Class Loss: {class_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Train CORAL Loss: {coral_loss_val:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.2e}")
            
            # Check improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                self.patience_counter = 0
                print("  🎯 New best validation accuracy!")
            else:
                self.patience_counter += 1
                print(f"  EarlyStopping counter: {self.patience_counter}/{self.patience}")
            
            print()
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"⚠️  Early stopping triggered at epoch {epoch}")
                break
        
        print(f"{'='*70}")
        print("✓ CORAL training complete!")
        print(f"  Best Val Acc: {self.best_val_acc:.2f}%")
        
        # Test
        print("\nEvaluating on X-ray test set...")
        test_acc = self.test()
        print(f"  Test Acc: {test_acc:.2f}%")
        
        # Save history
        history_path = self.results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n✓ Training history saved to {history_path}")
        
        # Save summary
        summary = {
            'method': 'CORAL (CORrelation ALignment)',
            'source_domain': 'CT',
            'target_domain': 'X-ray',
            'best_val_acc': self.best_val_acc,
            'test_acc': test_acc,
            'coral_weight': self.coral_weight,
            'num_epochs_trained': epoch,
            'learning_rate': self.learning_rate,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_path = self.results_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved to {summary_path}")
        print(f"{'='*70}\n")
        
        return self.best_val_acc, test_acc


def main():
    from pathlib import Path
    
    # Paths
    ct_checkpoint = '../checkpoints/stage1/best_model.pth'
    
    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets"
    ct_data_root = str(cache_root / "sampathlonka86" / "chestctscans" / "versions" / "1" / "Chest_CT")
    xray_data_root = str(cache_root / "prashant268" / "chest-xray-covid19-pneumonia" / "versions" / "2" / "Processed_XRay")
    
    # Train CORAL
    trainer = CORALTrainer(
        ct_checkpoint_path=ct_checkpoint,
        ct_data_root=ct_data_root,
        xray_data_root=xray_data_root,
        output_dir='.',
        learning_rate=1e-4,
        num_epochs=20,
        batch_size=32,
        coral_weight=1.0,  # Weight for CORAL loss
        device='cuda:1'
    )
    
    best_val_acc, test_acc = trainer.train()
    
    print("\n" + "="*70)
    print("CORAL RESULTS")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("="*70)


if __name__ == '__main__':
    main()
