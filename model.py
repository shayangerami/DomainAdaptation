"""
Model definition for COVID-19 classification with RadImageNet transfer learning.

This module provides:
- RadImageNet pretrained ResNet-50 model
- Binary classification head (COVID vs Non-COVID)
- Methods to freeze/unfreeze backbone for transfer learning
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
import os


class CovidClassifier(nn.Module):
    """
    COVID-19 binary classifier using RadImageNet pretrained ResNet-50.
    
    Architecture:
    - Backbone: ResNet-50 pretrained on RadImageNet
    - Head: Single linear layer (2048 → 2 classes)
    
    Args:
        pretrained_path: Path to RadImageNet weights (.pth file)
        num_classes: Number of output classes (default: 2 for binary)
        freeze_backbone: Whether to freeze backbone initially (default: True)
    """
    
    def __init__(
        self, 
        pretrained_path: Optional[str] = None,
        num_classes: int = 2,
        freeze_backbone: bool = True
    ):
        super(CovidClassifier, self).__init__()
        
        # Load ResNet-50 architecture
        self.model = models.resnet50(pretrained=False)
        
        # Load RadImageNet weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading RadImageNet weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different state dict formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Remove 'module.' prefix if present (from DataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # RadImageNet uses 'backbone.' prefix - map to standard ResNet names
            if any(k.startswith('backbone.') for k in state_dict.keys()):
                print("Detected RadImageNet format with 'backbone.' prefix, remapping...")
                new_state_dict = {}
                # Map backbone.N to standard ResNet layer names
                backbone_to_resnet = {
                    'backbone.0': 'conv1',
                    'backbone.1': 'bn1',
                    'backbone.4.': 'layer1.',
                    'backbone.5.': 'layer2.',
                    'backbone.6.': 'layer3.',
                    'backbone.7.': 'layer4.',
                }
                for k, v in state_dict.items():
                    new_k = k
                    for old_prefix, new_prefix in backbone_to_resnet.items():
                        if k.startswith(old_prefix):
                            new_k = k.replace(old_prefix, new_prefix)
                            break
                    new_state_dict[new_k] = v
                state_dict = new_state_dict
            
            # Load weights (excluding final fc layer)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and 'fc' not in k}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print(f"✓ Loaded {len(pretrained_dict)}/{len(model_dict)} layers from RadImageNet")
        else:
            if pretrained_path:
                print(f"Warning: RadImageNet weights not found at {pretrained_path}")
                print("Falling back to ImageNet weights...")
            else:
                print("No RadImageNet path provided, using ImageNet weights...")
            
            # Fallback to ImageNet pretrained weights
            self.model = models.resnet50(pretrained=True)
            print("✓ Loaded ImageNet pretrained weights")
        
        # Get the number of features in the final layer
        num_features = self.model.fc.in_features
        
        # Replace final classification layer (1000 → num_classes)
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Initialize new fc layer
        nn.init.xavier_uniform_(self.model.fc.weight)
        nn.init.zeros_(self.model.fc.bias)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
            print("✓ Backbone frozen - only training classifier")
        else:
            print("✓ Full model trainable")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def freeze_backbone(self):
        """Freeze early layers, unfreeze layer4 + classifier for better adaptation."""
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze layer4 (last ResNet block) - allows fine-grained adaptation
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        
        # Unfreeze final classifier
        for param in self.model.fc.parameters():
            param.requires_grad = True
        
        # Unfreeze all BatchNorm layers for better adaptation
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for param in module.parameters():
                    param.requires_grad = True
                module.train()
        
        print("Backbone frozen ❄️ (layer4 + BatchNorm + classifier trainable)")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        
        print("Backbone unfrozen 🔥")
    
    def get_trainable_params(self):
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_model(
    pretrained_path: Optional[str] = None,
    num_classes: int = 2,
    freeze_backbone: bool = True,
    device: str = 'cuda'
) -> CovidClassifier:
    """
    Factory function to create and initialize the model.
    
    Args:
        pretrained_path: Path to RadImageNet weights
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze backbone
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Initialized model on specified device
    """
    model = CovidClassifier(
        pretrained_path=pretrained_path,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone
    )
    
    model = model.to(device)
    
    total, trainable = model.count_parameters()
    print(f"\nModel Parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    
    return model


if __name__ == "__main__":
    """Test model creation."""
    
    print("=" * 70)
    print("Testing COVID-19 Classifier Model")
    print("=" * 70)
    
    # Test with frozen backbone (classifier training)
    print("\n[1] Testing with FROZEN backbone:")
    model = create_model(
        pretrained_path=None,  # Will use ImageNet as fallback
        num_classes=2,
        freeze_backbone=True,
        device='cpu'
    )
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test unfreezing
    print("\n[2] Testing UNFREEZING backbone:")
    model.unfreeze_backbone()
    total, trainable = model.count_parameters()
    print(f"  Trainable after unfreezing: {trainable:,} ({100*trainable/total:.2f}%)")
    
    print("\n" + "=" * 70)
    print("✓ Model tests passed!")
    print("=" * 70)
    
    print("\n💡 To use RadImageNet weights:")
    print("   1. Download weights from: https://github.com/BMEII-AI/RadImageNet")
    print("   2. Save as: /home/sgram/CTXray/weights/RadImageNet-ResNet50.pth")
    print("   3. Pass path to create_model(pretrained_path='...')")
