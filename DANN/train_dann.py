"""
Training script for DANN (Domain Adversarial Neural Network)

Trains on both CT (source) and X-ray (target) data simultaneously:
- CT data: has labels, trains label classifier
- X-ray data: has labels (supervised DANN), trains both classifiers
- Domain classifier: learns to distinguish CT vs X-ray
- Feature extractor: learns domain-invariant features via GRL
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

from dann_model import DANNModel, compute_lambda_alpha
from dataloader import create_dataloaders as create_ct_dataloaders
from xray_dataloader import create_xray_dataloaders


class DANNTrainer:
    """Trainer for Domain Adversarial Neural Network"""
    
    def __init__(self,
                 ct_checkpoint_path,
                 ct_data_root,
                 xray_data_root,
                 output_dir='.',
                 learning_rate=1e-4,
                 num_epochs=20,
                 batch_size=32,
                 device='cuda:1'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.results_dir = self.output_dir / 'results'
        
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n{'='*70}")
        print("Domain Adversarial Neural Network (DANN)")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        # Create DANN model
        self.model = DANNModel(
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
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.get_parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        # Loss functions
        self.label_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Early stopping
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0
        
        # History
        self.history = {
            'train_label_loss': [], 'train_domain_loss': [], 'train_total_loss': [],
            'train_label_acc': [], 'train_domain_acc': [],
            'val_label_acc': [], 'val_label_loss': [],
            'learning_rates': [], 'lambda_values': []
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
        """Train for one epoch with domain adversarial training"""
        self.model.train()
        
        # Compute GRL lambda (gradually increase adversarial strength)
        lambda_alpha = compute_lambda_alpha(epoch, self.num_epochs)
        
        # Track metrics
        total_label_loss = 0.0
        total_domain_loss = 0.0
        total_loss = 0.0
        label_correct = 0
        domain_correct = 0
        total_samples = 0
        
        # Create iterator for X-ray data (cycle if needed)
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
            
            xray_images = xray_images[:batch_size].to(self.device)  # Match batch size
            xray_labels = xray_labels[:batch_size].to(self.device)
            
            # Combine CT and X-ray data
            images = torch.cat([ct_images, xray_images], dim=0)
            labels = torch.cat([ct_labels, xray_labels], dim=0)
            
            # Domain labels: 0 for CT, 1 for X-ray
            domain_labels = torch.cat([
                torch.zeros(batch_size, dtype=torch.long),
                torch.ones(batch_size, dtype=torch.long)
            ]).to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                # Forward pass
                class_output, domain_output = self.model(images, alpha=lambda_alpha)
                
                # Label classification loss (both CT and X-ray have labels)
                label_loss = self.label_criterion(class_output, labels)
                
                # Domain classification loss
                domain_loss = self.domain_criterion(domain_output, domain_labels)
                
                # Total loss
                loss = label_loss + domain_loss
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Track metrics
            total_label_loss += label_loss.item() * images.size(0)
            total_domain_loss += domain_loss.item() * images.size(0)
            total_loss += loss.item() * images.size(0)
            
            _, label_preds = class_output.max(1)
            label_correct += label_preds.eq(labels).sum().item()
            
            _, domain_preds = domain_output.max(1)
            domain_correct += domain_preds.eq(domain_labels).sum().item()
            
            total_samples += images.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'L_loss': f'{label_loss.item():.4f}',
                'D_loss': f'{domain_loss.item():.4f}',
                'L_acc': f'{100.*label_correct/total_samples:.2f}%',
                'lambda': f'{lambda_alpha:.3f}'
            })
        
        # Compute epoch metrics
        epoch_label_loss = total_label_loss / total_samples
        epoch_domain_loss = total_domain_loss / total_samples
        epoch_total_loss = total_loss / total_samples
        epoch_label_acc = 100.0 * label_correct / total_samples
        epoch_domain_acc = 100.0 * domain_correct / total_samples
        
        return epoch_label_loss, epoch_domain_loss, epoch_total_loss, epoch_label_acc, epoch_domain_acc, lambda_alpha
    
    @torch.no_grad()
    def validate(self):
        """Validate on X-ray validation set (target domain)"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.xray_val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Only use label classifier for validation
            class_output, _ = self.model(images, alpha=0.0)
            loss = self.label_criterion(class_output, labels)
            
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
            
            class_output, _ = self.model(images, alpha=0.0)
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
        print("Starting DANN training...")
        print(f"{'='*70}\n")
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"Epoch {epoch}/{self.num_epochs}")
            print("-" * 70)
            
            # Train
            label_loss, domain_loss, total_loss, label_acc, domain_acc, lambda_val = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_label_loss'].append(label_loss)
            self.history['train_domain_loss'].append(domain_loss)
            self.history['train_total_loss'].append(total_loss)
            self.history['train_label_acc'].append(label_acc)
            self.history['train_domain_acc'].append(domain_acc)
            self.history['val_label_loss'].append(val_loss)
            self.history['val_label_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            self.history['lambda_values'].append(lambda_val)
            
            # Print metrics
            print(f"  Train Label Loss: {label_loss:.4f} | Label Acc: {label_acc:.2f}%")
            print(f"  Train Domain Loss: {domain_loss:.4f} | Domain Acc: {domain_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.2e} | Lambda: {lambda_val:.3f}")
            
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
        print("✓ DANN training complete!")
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
            'method': 'DANN (Domain Adversarial Neural Network)',
            'source_domain': 'CT',
            'target_domain': 'X-ray',
            'best_val_acc': self.best_val_acc,
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
    from pathlib import Path
    
    # Paths
    ct_checkpoint = '../checkpoints/stage1/best_model.pth'
    
    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets"
    ct_data_root = str(cache_root / "sampathlonka86" / "chestctscans" / "versions" / "1" / "Chest_CT")
    xray_data_root = str(cache_root / "prashant268" / "chest-xray-covid19-pneumonia" / "versions" / "2" / "Processed_XRay")
    
    # Train DANN
    trainer = DANNTrainer(
        ct_checkpoint_path=ct_checkpoint,
        ct_data_root=ct_data_root,
        xray_data_root=xray_data_root,
        output_dir='.',
        learning_rate=5e-5,  # Lower LR for adversarial training
        num_epochs=20,
        batch_size=32,
        device='cuda:1'
    )
    
    best_val_acc, test_acc = trainer.train()
    
    print("\n" + "="*70)
    print("DANN RESULTS")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("="*70)


if __name__ == '__main__':
    main()
