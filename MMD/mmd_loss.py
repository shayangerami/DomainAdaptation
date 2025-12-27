"""
MMD (Maximum Mean Discrepancy) Loss

Reference: "Learning Transferable Features with Deep Adaptation Networks" (Long et al., 2015)
https://arxiv.org/abs/1502.02791

MMD measures the distance between two distributions using kernel embeddings.
"""

import torch
import torch.nn as nn
import numpy as np


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Compute Gaussian (RBF) kernel matrix between source and target.
    
    Uses multiple bandwidths to capture different scales.
    
    Args:
        source: (n_source, feature_dim) tensor
        target: (n_target, feature_dim) tensor
        kernel_mul: Multiplier for bandwidth
        kernel_num: Number of different bandwidths
        fix_sigma: Fixed bandwidth (if None, auto-compute)
    
    Returns:
        kernel_matrix: (n_source+n_target, n_source+n_target) kernel values
    """
    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)
    
    # Compute pairwise L2 distances
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    
    # Compute bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    
    # Compute multiple bandwidths
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    
    # Compute kernel matrix with multiple bandwidths
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    
    return sum(kernel_val)  # Sum over all bandwidths


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Compute Maximum Mean Discrepancy (MMD) between source and target.
    
    MMD² = E[k(x_s, x_s)] - 2*E[k(x_s, x_t)] + E[k(x_t, x_t)]
    
    Where:
    - k: Gaussian kernel
    - x_s: Source features
    - x_t: Target features
    - E: Expected value (mean)
    
    Args:
        source: (batch_size, feature_dim) tensor from source domain
        target: (batch_size, feature_dim) tensor from target domain
        kernel_mul: Kernel bandwidth multiplier
        kernel_num: Number of kernels
        fix_sigma: Fixed bandwidth
    
    Returns:
        mmd_loss: Scalar MMD² value
    """
    batch_size = source.size(0)
    
    # Compute kernel matrix
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    
    # Split kernel matrix into 4 blocks
    # [K_ss  K_st]
    # [K_ts  K_tt]
    XX = kernels[:batch_size, :batch_size]          # Source-Source
    YY = kernels[batch_size:, batch_size:]          # Target-Target
    XY = kernels[:batch_size, batch_size:]          # Source-Target
    YX = kernels[batch_size:, :batch_size]          # Target-Source
    
    # MMD² = mean(K_ss) - 2*mean(K_st) + mean(K_tt)
    loss = torch.mean(XX) - torch.mean(XY) - torch.mean(YX) + torch.mean(YY)
    
    return loss


class MMDModel(nn.Module):
    """
    Model with MMD loss for domain adaptation.
    
    Uses a pre-trained model and adds MMD loss to minimize domain discrepancy.
    """
    
    def __init__(self, ct_checkpoint_path, num_classes=2, device='cuda'):
        super(MMDModel, self).__init__()
        
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
        
        print("✓ MMD model initialized")
        print(f"  - Feature Extractor: ResNet-50")
        print(f"  - Classifier: {num_classes} classes")
        print(f"  - MMD loss: Minimizes distribution distance via kernel embedding")
    
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
