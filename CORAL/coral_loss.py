"""
CORAL (CORrelation ALignment) Loss

Reference: "Return of Frustratingly Easy Domain Adaptation" (Sun et al., 2016)
https://arxiv.org/abs/1511.05547

CORAL aligns the second-order statistics (covariance) between source and target domains.
"""

import torch
import torch.nn as nn


def coral_loss(source_features, target_features):
    """
    Compute CORAL loss between source and target features.
    
    CORAL Loss = || C_s - C_t ||_F^2 / (4 * d^2)
    
    Where:
    - C_s: covariance matrix of source features
    - C_t: covariance matrix of target features
    - ||.||_F: Frobenius norm
    - d: feature dimension
    
    Args:
        source_features: (batch_size, feature_dim) tensor from source domain
        target_features: (batch_size, feature_dim) tensor from target domain
    
    Returns:
        coral_loss: scalar loss value
    """
    d = source_features.size(1)  # feature dimension
    
    # Compute covariance matrices
    # Center the features (subtract mean)
    source_mean = source_features.mean(0, keepdim=True)
    target_mean = target_features.mean(0, keepdim=True)
    
    source_centered = source_features - source_mean
    target_centered = target_features - target_mean
    
    # Covariance = (X^T * X) / (n - 1)
    ns = source_features.size(0)
    nt = target_features.size(0)
    
    cov_source = (source_centered.t() @ source_centered) / (ns - 1)
    cov_target = (target_centered.t() @ target_centered) / (nt - 1)
    
    # Frobenius norm of difference
    loss = torch.norm(cov_source - cov_target, p='fro') ** 2
    
    # Normalize by 4 * d^2 (as in paper)
    loss = loss / (4 * d * d)
    
    return loss


class CoralModel(nn.Module):
    """
    Model with CORAL loss for domain adaptation.
    
    Uses a pre-trained model and adds CORAL loss to align feature distributions.
    """
    
    def __init__(self, ct_checkpoint_path, num_classes=2, device='cuda'):
        super(CoralModel, self).__init__()
        
        self.device = device
        
        # Load pre-trained CT model
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        print(f"Loading CT checkpoint: {ct_checkpoint_path}")
        checkpoint = torch.load(ct_checkpoint_path, map_location=device)
        
        from model import create_model
        base_model = create_model(
            pretrained_path=None,
            num_classes=num_classes,
            freeze_backbone=False,
            device=device
        )
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Extract feature extractor (ResNet without final FC)
        self.feature_extractor = nn.Sequential(*list(base_model.model.children())[:-1])
        
        # Classifier (reuse from CT model)
        self.classifier = base_model.model.fc
        
        print("✓ CORAL model initialized")
        print(f"  - Feature Extractor: ResNet-50")
        print(f"  - Classifier: {num_classes} classes")
        print(f"  - CORAL loss: Aligns covariance matrices")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images
        
        Returns:
            features: Feature representation (2048-dim)
            class_output: Classification logits
        """
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Classify
        class_output = self.classifier(features)
        
        return features, class_output
    
    def get_parameters(self):
        """Get parameter groups for optimizer"""
        return [
            {'params': self.feature_extractor.parameters()},
            {'params': self.classifier.parameters()}
        ]
