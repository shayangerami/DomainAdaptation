"""
Training script for COVID-19 classification on CT scans.

Stage 1: Train classifier only (frozen backbone) - 5-10 epochs
Stage 2: Fine-tune full model (unfrozen backbone) - optional

Usage:
    # Stage 1: Train classifier with frozen backbone
    python train.py --stage 1
    
    # Stage 2: Fine-tune full model  
    python train.py --stage 2
    
    # Custom epochs
    python train.py --stage 1 --epochs 8
    
    # Resume from checkpoint
    python train.py --stage 1 --resume checkpoints/stage1/best_model.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import argparse
import time
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from model import create_model
from dataloader import create_dataloaders, get_class_weights
from config import get_config, print_config


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class Trainer:
    """Trainer class for COVID-19 classification."""
    
    def __init__(self, config, stage='stage1'):
        self.config = config
        self.stage = stage
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*70}")
        print(f"Initializing Trainer - {config['name']}")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        
        # Create dataloaders
        print("\nLoading data...")
        from config import CT_DATA_PATH
        self.train_loader, self.val_loader = create_dataloaders(
            data_root=str(CT_DATA_PATH),
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            combine_test_val=False
        )
        
        print(f"✓ Train: {len(self.train_loader.dataset)} samples")
        print(f"✓ Val: {len(self.val_loader.dataset)} samples")
        
        # Create model
        print("\nInitializing model...")
        self.model = create_model(
            pretrained_path=config.get('pretrained_path'),
            num_classes=config['num_classes'],
            freeze_backbone=config['freeze_backbone'],
            device=self.device
        )
        
        # Loss function with class weights
        if config['use_class_weights']:
            class_weights = get_class_weights(self.train_loader.dataset)
            class_weights = class_weights.to(self.device)
            print(f"\n✓ Using class weights: {class_weights.tolist()}")
        else:
            class_weights = None
        
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.get('label_smoothing', 0.0)
        )
        
        # Optimizer
        trainable_params = self.model.get_trainable_params()
        if config['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(
                trainable_params,
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                trainable_params,
                lr=config['learning_rate'],
                momentum=config.get('momentum', 0.9),
                weight_decay=config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")
        
        print(f"✓ Optimizer: {config['optimizer'].upper()}")
        print(f"✓ Learning rate: {config['learning_rate']}")
        
        # Scheduler
        self.scheduler = None
        if config.get('scheduler'):
            if config['scheduler'] == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config['scheduler_params']['T_max'],
                    eta_min=config['scheduler_params'].get('eta_min', 0)
                )
            elif config['scheduler'] == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=config['scheduler_params']['step_size'],
                    gamma=config['scheduler_params']['gamma']
                )
            print(f"✓ Scheduler: {config['scheduler']}")
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', False) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("✓ Mixed precision training enabled")
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 5),
            min_delta=config.get('min_delta', 0.001)
        )
        
        # Tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.train_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': []
        }
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"✓ Checkpoint directory: {self.checkpoint_dir}")
        print(f"{'='*70}\n")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]} [Train]')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            if (batch_idx + 1) % self.config.get('log_interval', 10) == 0:
                pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]} [Val]  ')
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'config': self.config,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_model.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model to {best_path}")
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training...")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.2e}")
            
            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                print(f"  🎯 New best validation accuracy!")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                break
            
            print(f"{'='*70}")
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n✓ Training complete!")
        print(f"  Time elapsed: {elapsed_time/60:.2f} minutes")
        print(f"  Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"  Best Val Loss: {self.best_val_loss:.4f}")
        
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        print(f"\n✓ History saved to: {history_path}")
        print(f"{'='*70}")
        
        return self.train_history


def main():
    parser = argparse.ArgumentParser(description='Train COVID-19 classifier')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                        help='Training stage (1: classifier only, 2: fine-tune all)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Get configuration
    stage_name = f'stage{args.stage}'
    config = get_config(stage_name)
    
    # Override config with command-line args
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.device:
        config['device'] = args.device
    
    # Print configuration
    print_config(config)
    
    # Set random seed for reproducibility
    if config.get('deterministic'):
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        np.random.seed(config['seed'])
        print(f"\n✓ Random seed set to {config['seed']}")
    
    # Create trainer
    trainer = Trainer(config, stage=stage_name)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✓ Checkpoint loaded successfully")
    
    # Load Stage 1 weights for Stage 2
    elif args.stage == 2 and config.get('load_from_stage1'):
        stage1_checkpoint = Path(config['checkpoint_dir']).parent / 'stage1' / 'best_model.pth'
        if stage1_checkpoint.exists():
            print(f"\nLoading Stage 1 weights from: {stage1_checkpoint}")
            checkpoint = torch.load(stage1_checkpoint, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Stage 1 weights loaded successfully")
        else:
            print(f"⚠️  Warning: Stage 1 checkpoint not found at {stage1_checkpoint}")
            print("    Starting Stage 2 from scratch...")
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Latest checkpoint saved. You can resume with --resume")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
