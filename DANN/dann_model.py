"""
Domain Adversarial Neural Network (DANN) for CT → X-ray adaptation

Reference: "Domain-Adversarial Training of Neural Networks" (Ganin et al., 2016)
https://arxiv.org/abs/1505.07818
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL)
    
    Forward pass: identity function
    Backward pass: multiply gradient by -lambda
    
    This makes the domain classifier try to distinguish domains,
    while the feature extractor learns to confuse it.
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer wrapper"""
    
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainClassifier(nn.Module):
    """
    Domain classifier: distinguishes CT from X-ray
    
    Takes features from backbone and predicts domain (0=CT, 1=X-ray)
    """
    
    def __init__(self, input_dim=2048, hidden_dim=1024):
        super(DomainClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2)  # 2 domains: CT, X-ray
        )
    
    def forward(self, x):
        return self.classifier(x)


class DANNModel(nn.Module):
    """
    Domain Adversarial Neural Network
    
    Components:
    1. Feature Extractor: ResNet-50 backbone (shared)
    2. Label Classifier: COVID vs Normal (task-specific)
    3. Domain Classifier: CT vs X-ray (adversarial)
    
    Training:
    - Label classifier tries to correctly classify COVID/Normal
    - Domain classifier tries to distinguish CT/X-ray
    - Feature extractor tries to fool domain classifier (via GRL)
      while helping label classifier
    
    This encourages learning domain-invariant features.
    """
    
    def __init__(self, ct_checkpoint_path, num_classes=2, device='cuda'):
        super(DANNModel, self).__init__()
        
        self.device = device
        
        # Load pre-trained CT model as feature extractor
        print(f"Loading CT checkpoint: {ct_checkpoint_path}")
        checkpoint = torch.load(ct_checkpoint_path, map_location=device)
        
        # Create base model
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
        
        # Label classifier (reuse from CT model)
        self.label_classifier = base_model.model.fc
        
        # Gradient reversal layer
        self.grl = GradientReversalLayer(lambda_=1.0)
        
        # Domain classifier
        self.domain_classifier = DomainClassifier(
            input_dim=2048,  # ResNet-50 feature dim
            hidden_dim=1024
        )
        
        print("✓ DANN model initialized")
        print(f"  - Feature Extractor: ResNet-50")
        print(f"  - Label Classifier: {num_classes} classes")
        print(f"  - Domain Classifier: 2 domains (CT, X-ray)")
    
    def forward(self, x, alpha=1.0):
        """
        Forward pass
        
        Args:
            x: Input images
            alpha: GRL lambda parameter (controls adversarial strength)
        
        Returns:
            class_output: Label predictions (COVID/Normal)
            domain_output: Domain predictions (CT/X-ray)
        """
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Label classification (for task)
        class_output = self.label_classifier(features)
        
        # Domain classification (adversarial)
        self.grl.lambda_ = alpha  # Update GRL strength
        reversed_features = self.grl(features)
        domain_output = self.domain_classifier(reversed_features)
        
        return class_output, domain_output
    
    def get_parameters(self):
        """Get parameter groups for optimizer"""
        return [
            {'params': self.feature_extractor.parameters()},
            {'params': self.label_classifier.parameters()},
            {'params': self.domain_classifier.parameters()}
        ]


def compute_lambda_alpha(epoch, total_epochs):
    """
    Compute GRL lambda schedule (gradually increase adversarial strength)
    
    Following DANN paper: lambda_p = 2 / (1 + exp(-10 * p)) - 1
    where p = epoch / total_epochs
    """
    p = float(epoch) / float(total_epochs)
    lambda_alpha = 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0
    return lambda_alpha.item()
