"""
Test script to visualize samples from the CT scan dataset
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataloader import create_dataloaders


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for visualization"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def visualize_batch(images, labels, class_names=['Non-COVID', 'COVID']):
    """Visualize a batch of images"""
    batch_size = images.shape[0]
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx in range(batch_size):
        img = denormalize(images[idx])
        img = img.permute(1, 2, 0).numpy()
        
        axes[idx].imshow(img)
        axes[idx].set_title(f'{class_names[labels[idx].item()]}', fontsize=12)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(batch_size, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Loading dataloaders...")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root="/home/sgram/.cache/kagglehub/datasets/sampathlonka86/chestctscans/versions/1/Chest_CT",
        batch_size=16,
        img_size=224,
        num_workers=2
    )
    
    # Get a batch from train set
    images, labels = next(iter(train_loader))
    
    print(f"\nVisualizing {images.shape[0]} samples from training set...")
    fig = visualize_batch(images, labels)
    
    # Save figure
    output_path = "/home/sgram/CTXray/results/sample_batch.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    print("\nDataset Summary:")
    print("=" * 60)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"  - COVID: {train_loader.dataset.get_class_distribution()['COVID_CT']}")
    print(f"  - Non-COVID: {train_loader.dataset.get_class_distribution()['NONCOVID_CT']}")
    print(f"\nValidation samples: {len(val_loader.dataset)}")
    print(f"  - COVID: {val_loader.dataset.get_class_distribution()['COVID_CT']}")
    print(f"  - Non-COVID: {val_loader.dataset.get_class_distribution()['NONCOVID_CT']}")
    print(f"\nTest samples: {len(test_loader.dataset)}")
    print(f"  - COVID: {test_loader.dataset.get_class_distribution()['COVID_CT']}")
    print(f"  - Non-COVID: {test_loader.dataset.get_class_distribution()['NONCOVID_CT']}")
    print("=" * 60)
