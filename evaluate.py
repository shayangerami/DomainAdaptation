"""
Evaluation script for COVID-19 classification model.

Computes comprehensive metrics:
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC with ROC curve visualization  
- Confusion matrix
- Per-class performance
- Domain gap analysis (CT vs X-ray)

Usage:
    # Evaluate on CT test set
    python evaluate.py --checkpoint checkpoints/stage1/best_model.pth
    
    # Evaluate on X-ray test set (domain gap)
    python evaluate.py --checkpoint checkpoints/stage1/best_model.pth --domain xray
    
    # Save predictions
    python evaluate.py --checkpoint checkpoints/stage1/best_model.pth --save-predictions
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from model import create_model
from dataloader import create_dataloaders
from xray_dataloader import create_xray_dataloaders
from config import RESULTS_DIR, DATA_CONFIG


class Evaluator:
    """Evaluator class for COVID-19 classification."""
    
    def __init__(self, checkpoint_path, domain='ct', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.domain = domain
        
        print(f"\n{'='*70}")
        print(f"Initializing Evaluator")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Domain: {domain.upper()}")
        
        # Load checkpoint
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        config = checkpoint.get('config', {})
        self.model = create_model(
            pretrained_path=None,  # Already trained
            num_classes=2,
            freeze_backbone=False,  # Eval mode doesn't need freezing
            device=self.device
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("✓ Model loaded successfully")
        
        # Load data
        print(f"\nLoading {domain.upper()} data...")
        from config import CT_DATA_PATH, XRAY_DATA_PATH
        if domain == 'ct':
            _, _, self.test_loader = create_dataloaders(
                data_root=str(CT_DATA_PATH),
                batch_size=32,
                num_workers=4
            )
            self.class_names = DATA_CONFIG['class_names']
        elif domain == 'xray':
            _, _, self.test_loader = create_xray_dataloaders(
                data_root=str(XRAY_DATA_PATH),
                batch_size=32,
                num_workers=4
            )
            self.class_names = DATA_CONFIG['xray_class_names']
        else:
            raise ValueError(f"Unknown domain: {domain}")
        
        print(f"✓ Test set: {len(self.test_loader.dataset)} samples")
        
        # Results directory
        self.results_dir = RESULTS_DIR / f'eval_{domain}'
        self.results_dir.mkdir(exist_ok=True, parents=True)
        print(f"✓ Results will be saved to: {self.results_dir}")
        
        print(f"{'='*70}\n")
    
    @torch.no_grad()
    def predict(self):
        """Run inference and collect predictions."""
        print("Running inference...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in tqdm(self.test_loader, desc='Evaluating'):
            images = images.to(self.device)
            
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def compute_metrics(self, labels, preds, probs):
        """Compute all evaluation metrics."""
        print("\nComputing metrics...")
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['precision'] = precision_score(labels, preds, average='weighted')
        metrics['recall'] = recall_score(labels, preds, average='weighted')
        metrics['f1_score'] = f1_score(labels, preds, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(labels, preds, average=None)
        recall_per_class = recall_score(labels, preds, average=None)
        f1_per_class = f1_score(labels, preds, average=None)
        
        metrics['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
            }
        
        # AUC-ROC (for positive class)
        if probs.shape[1] == 2:
            metrics['auc_roc'] = roc_auc_score(labels, probs[:, 1])
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(labels, preds).tolist()
        
        # Classification report
        report = classification_report(
            labels, preds,
            target_names=self.class_names,
            output_dict=True
        )
        metrics['classification_report'] = report
        
        return metrics
    
    def plot_confusion_matrix(self, labels, preds):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix - {self.domain.upper()} Domain')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = self.results_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    def plot_roc_curve(self, labels, probs):
        """Plot and save ROC curve."""
        if probs.shape[1] != 2:
            print("⚠️  ROC curve only available for binary classification")
            return
        
        fpr, tpr, thresholds = roc_curve(labels, probs[:, 1])
        auc = roc_auc_score(labels, probs[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.domain.upper()} Domain')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = self.results_dir / 'roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curve saved to: {save_path}")
    
    def plot_per_class_metrics(self, metrics):
        """Plot per-class performance."""
        per_class = metrics['per_class']
        
        classes = list(per_class.keys())
        precision = [per_class[c]['precision'] for c in classes]
        recall = [per_class[c]['recall'] for c in classes]
        f1 = [per_class[c]['f1_score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title(f'Per-Class Performance - {self.domain.upper()} Domain')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = self.results_dir / 'per_class_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Per-class metrics saved to: {save_path}")
    
    def save_predictions(self, labels, preds, probs):
        """Save predictions to file."""
        predictions = {
            'labels': labels.tolist(),
            'predictions': preds.tolist(),
            'probabilities': probs.tolist(),
            'class_names': self.class_names,
        }
        
        save_path = self.results_dir / 'predictions.json'
        with open(save_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"✓ Predictions saved to: {save_path}")
    
    def print_metrics(self, metrics):
        """Print metrics in a nice format."""
        print("\n" + "="*70)
        print(f"EVALUATION RESULTS - {self.domain.upper()} Domain")
        print("="*70)
        
        print("\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if 'auc_roc' in metrics:
            print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        print("\nPer-Class Metrics:")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"\n  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.4f}")
            print(f"    Recall:    {class_metrics['recall']:.4f}")
            print(f"    F1-Score:  {class_metrics['f1_score']:.4f}")
        
        print("\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"               Predicted")
        print(f"               {self.class_names[0]:<15} {self.class_names[1]}")
        print(f"  Actual")
        print(f"  {self.class_names[0]:<13} {cm[0,0]:<15} {cm[0,1]}")
        print(f"  {self.class_names[1]:<13} {cm[1,0]:<15} {cm[1,1]}")
        
        print("\n" + "="*70)
    
    def evaluate(self, save_predictions=True):
        """Run full evaluation pipeline."""
        # Get predictions
        labels, preds, probs = self.predict()
        
        # Compute metrics
        metrics = self.compute_metrics(labels, preds, probs)
        
        # Print results
        self.print_metrics(metrics)
        
        # Save metrics
        metrics_path = self.results_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Metrics saved to: {metrics_path}")
        
        # Create visualizations
        print("\nGenerating visualizations...")
        self.plot_confusion_matrix(labels, preds)
        self.plot_roc_curve(labels, probs)
        self.plot_per_class_metrics(metrics)
        
        # Save predictions
        if save_predictions:
            self.save_predictions(labels, preds, probs)
        
        print(f"\n✓ Evaluation complete! Results saved to: {self.results_dir}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate COVID-19 classifier')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--domain', type=str, default='ct', choices=['ct', 'xray'],
                        help='Domain to evaluate on (ct or xray)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions to file')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        return
    
    # Create evaluator
    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        domain=args.domain,
        device=args.device
    )
    
    # Run evaluation
    try:
        metrics = evaluator.evaluate(save_predictions=args.save_predictions)
        
        # Domain gap warning
        if args.domain == 'xray':
            print("\n" + "⚠️ " * 20)
            print("DOMAIN GAP ANALYSIS")
            print("⚠️ " * 20)
            print("\nThis model was trained on CT scans but evaluated on X-rays.")
            print("The performance drop indicates the domain shift between CT and X-ray.")
            print("To improve X-ray performance:")
            print("  1. Use domain adaptation techniques (DANN, CORAL, MMD)")
            print("  2. Fine-tune on X-ray data")
            print("  3. Use domain-adversarial training")
            print("⚠️ " * 20)
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
