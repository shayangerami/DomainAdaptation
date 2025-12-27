"""
Visualize samples from both CT and X-ray datasets side by side
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataloader import create_dataloaders
from xray_dataloader import create_xray_dataloaders


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for visualization"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def visualize_datasets():
    """Create side-by-side visualization of CT and X-ray samples"""
    
    print("Loading datasets...")
    
    # Load CT dataloaders
    ct_train, _, _ = create_dataloaders(
        data_root="/home/sgram/.cache/kagglehub/datasets/sampathlonka86/chestctscans/versions/1/Chest_CT",
        batch_size=8,
        img_size=224,
        num_workers=2
    )
    
    # Load X-ray dataloaders
    xray_train, _, _ = create_xray_dataloaders(
        data_root="/home/sgram/.cache/kagglehub/datasets/prashant268/chest-xray-covid19-pneumonia/versions/2/Processed_XRay",
        batch_size=8,
        img_size=224,
        num_workers=2
    )
    
    # Get samples
    ct_images, ct_labels = next(iter(ct_train))
    xray_images, xray_labels = next(iter(xray_train))
    
    # Create visualization
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Domain Adaptation: CT Scans (Source) vs X-Rays (Target)', fontsize=16, fontweight='bold')
    
    # CT samples (top 2 rows)
    for idx in range(8):
        row = idx // 4
        col = idx % 4
        
        img = denormalize(ct_images[idx])
        img = img.permute(1, 2, 0).numpy()
        
        axes[row, col].imshow(img)
        label_name = 'COVID' if ct_labels[idx].item() == 1 else 'Non-COVID'
        axes[row, col].set_title(f'CT: {label_name}', fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    # X-ray samples (bottom 2 rows)
    for idx in range(8):
        row = idx // 4 + 2
        col = idx % 4
        
        img = denormalize(xray_images[idx])
        img = img.permute(1, 2, 0).numpy()
        
        axes[row, col].imshow(img)
        label_name = 'COVID-19' if xray_labels[idx].item() == 1 else 'Normal'
        axes[row, col].set_title(f'X-Ray: {label_name}', fontsize=12, fontweight='bold', color='darkblue')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = "/home/sgram/CTXray/results/domain_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    # Create statistics summary
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axis('off')
    
    summary_text = """
    DOMAIN ADAPTATION SETUP SUMMARY
    ================================
    
    SOURCE DOMAIN: CT Scans
    -----------------------
    • Total Images: 846
    • Training: 586 (269 COVID, 317 Non-COVID)
    • Validation: 60 (30 COVID, 30 Non-COVID)
    • Test: 200 (100 COVID, 100 Non-COVID)
    • Modality: 3D volumetric scans
    • Class Balance: Slightly imbalanced (54% vs 46%)
    
    TARGET DOMAIN: Chest X-Rays
    ---------------------------
    • Total Images: 920
    • Training: 644 (322 COVID-19, 322 Normal)
    • Validation: 138 (69 COVID-19, 69 Normal)
    • Test: 138 (69 COVID-19, 69 Normal)
    • Modality: 2D projection images
    • Class Balance: Perfectly balanced (50% vs 50%)
    
    DOMAIN SHIFT CHALLENGES
    -----------------------
    ✗ Different imaging modalities (3D→2D)
    ✗ Different resolutions and contrasts
    ✗ Different anatomical representations
    ✗ Distribution shift between domains
    
    ADAPTATION STRATEGIES
    ---------------------
    ○ Feature-level: DANN, CORAL, MMD
    ○ Pixel-level: CycleGAN, style transfer
    ○ Semi-supervised: Pseudo-labeling
    ○ Multi-stage: Pretrain → Adapt → Fine-tune
    
    STATUS: ✓ Both datasets ready!
    """
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    summary_path = "/home/sgram/CTXray/results/setup_summary.png"
    fig2.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary to: {summary_path}")
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  1. {output_path}")
    print(f"  2. {summary_path}")
    print("\nDatasets are ready for domain adaptation experiments!")
    print("=" * 70)


if __name__ == "__main__":
    visualize_datasets()
