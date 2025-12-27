"""
Fine-tuning script for domain adaptation: CT → X-ray

Loads CT-trained model and fine-tunes on X-ray dataset.
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

from model import create_model
from xray_dataloader import create_xray_dataloaders


class FineTuner:
    """Fine-tuning trainer for X-ray domain adaptation"""
    
    def __init__(self, 
                 ct_checkpoint_path,
                 xray_data_root,
                 output_dir='FineTuned',
                 learning_rate=1e-4,
                 num_epochs=15,
                 batch_size=32,
                 device='cuda:1'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.results_dir = self.output_dir / 'results'
        
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n{'='*70}")
        print("Fine-tuning: CT Model → X-ray Domain")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        # Load CT-trained model
        print(f"\nLoading CT checkpoint: {ct_checkpoint_path}")
        checkpoint = torch.load(ct_checkpoint_path, map_location=self.device)
        
        self.model = create_model(
            pretrained_path=None,
            num_classes=2,
            freeze_backbone=False,  # Fine-tune all layers
            device=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ CT model loaded successfully")
        
        # Load X-ray data
        print(f"\nLoading X-ray data from: {xray_data_root}")
        self.train_loader, self.val_loader, self.test_loader = create_xray_dataloaders(
            data_root=xray_data_root,
            batch_size=batch_size,
            num_workers=4
        )
        
        # Training setup
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Use lower learning rate for fine-tuning
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        # Loss and mixed precision
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()
        
        # Early stopping
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        
        # History tracking
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        
        print(f"\n✓ Optimizer: Adam (lr={learning_rate})")
        print(f"✓ Scheduler: CosineAnnealing (T_max={num_epochs})")
        print(f"✓ Training samples: {len(self.train_loader.dataset)}")
        print(f"✓ Validation samples: {len(self.val_loader.dataset)}")
        print(f"✓ Test samples: {len(self.test_loader.dataset)}")
        print(f"{'='*70}\n")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / total
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    @torch.no_grad()
    def test(self):
        """Test on test set"""
        self.model.eval()
        correct = 0
        total = 0
        
        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        return test_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, path)
            print(f"    ✓ Saved best model to {path}")
    
    def train(self):
        """Main training loop"""
        print("Starting fine-tuning...")
        print(f"{'='*70}\n")
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"Epoch {epoch}/{self.num_epochs}")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Print metrics
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.2e}")
            
            # Check for improvement
            improved = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                improved = True
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
        print("✓ Fine-tuning complete!")
        print(f"  Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"  Best Val Loss: {self.best_val_loss:.4f}")
        
        # Test on test set
        print("\nEvaluating on test set...")
        test_acc = self.test()
        print(f"  Test Acc: {test_acc:.2f}%")
        
        # Save history
        history_path = self.results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n✓ Training history saved to {history_path}")
        
        # Save summary
        summary = {
            'method': 'Fine-tuning',
            'source_domain': 'CT',
            'target_domain': 'X-ray',
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'test_acc': test_acc,
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
    # Paths
    from pathlib import Path
    ct_checkpoint = '../checkpoints/stage1/best_model.pth'
    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets"
    xray_data_root = str(cache_root / "prashant268" / "chest-xray-covid19-pneumonia" / "versions" / "2" / "Processed_XRay")
    
    # Fine-tune
    finetuner = FineTuner(
        ct_checkpoint_path=ct_checkpoint,
        xray_data_root=xray_data_root,
        output_dir='.',  # Current dir is FineTuned/
        learning_rate=1e-4,  # Lower LR for fine-tuning
        num_epochs=15,
        batch_size=32,
        device='cuda:1'
    )
    
    best_val_acc, test_acc = finetuner.train()
    
    print("\n" + "="*70)
    print("FINE-TUNING RESULTS")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("="*70)


if __name__ == '__main__':
    main()
