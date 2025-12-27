"""
Configuration file for COVID-19 classification training.

Contains all hyperparameters and paths for:
- Stage 1: Frozen backbone, train classifier only
- Stage 2: Unfreeze backbone, fine-tune full model
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Project root
PROJECT_ROOT = Path(__file__).parent
CACHE_ROOT = Path.home() / ".cache" / "kagglehub" / "datasets"

# Data paths (in kagglehub cache)
CT_DATA_PATH = CACHE_ROOT / "sampathlonka86" / "chestctscans" / "versions" / "1" / "Chest_CT"
XRAY_DATA_PATH = CACHE_ROOT / "prashant268" / "chest-xray-covid19-pneumonia" / "versions" / "2" / "Processed_XRay"

# Model weights
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RADIMAGENET_WEIGHTS = WEIGHTS_DIR / "RadImageNet-ResNet50.pth"

# Output directories
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [WEIGHTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    'architecture': 'resnet50',
    'num_classes': 2,
    'pretrained_path': str(RADIMAGENET_WEIGHTS) if RADIMAGENET_WEIGHTS.exists() else None,
}


# ============================================================================
# STAGE 1: CLASSIFIER TRAINING (Frozen Backbone)
# ============================================================================

STAGE1_CONFIG = {
    'name': 'stage1_classifier',
    'description': 'Train classifier with layer4 unfrozen',
    
    # Training
    'num_epochs': 20,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'optimizer': 'adam',
    
    # Model
    'freeze_backbone': True,
    
    # Scheduler
    'scheduler': 'cosine',
    'scheduler_params': {
        'T_max': 20,
        'eta_min': 1e-6,
    },
    
    # Early stopping
    'patience': 10,
    'min_delta': 0.001,
    
    # Checkpointing
    'save_best_only': True,
    'checkpoint_dir': CHECKPOINTS_DIR / 'stage1',
}


# ============================================================================
# STAGE 2: FULL MODEL FINE-TUNING (Unfrozen Backbone)
# ============================================================================

STAGE2_CONFIG = {
    'name': 'stage2_finetune',
    'description': 'Fine-tune full model with unfrozen backbone',
    
    # Training
    'num_epochs': 20,
    'batch_size': 16,  # Smaller batch for stability
    'learning_rate': 1e-5,  # Much lower LR for fine-tuning
    'weight_decay': 1e-4,
    'optimizer': 'sgd',
    'momentum': 0.9,
    
    # Model
    'freeze_backbone': False,
    'load_from_stage1': True,  # Load best checkpoint from stage 1
    
    # Scheduler
    'scheduler': 'step',
    'scheduler_params': {
        'step_size': 7,
        'gamma': 0.1,
    },
    
    # Early stopping
    'patience': 7,
    'min_delta': 0.0005,
    
    # Checkpointing
    'save_best_only': True,
    'checkpoint_dir': CHECKPOINTS_DIR / 'stage2',
}


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

DATA_CONFIG = {
    # Image preprocessing
    'image_size': 224,
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
    
    # Data loading
    'num_workers': 4,
    'pin_memory': True,
    
    # Class names
    'class_names': ['Non-COVID', 'COVID'],  # For CT scans
    'xray_class_names': ['Normal', 'COVID-19'],  # For X-rays
}


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAIN_CONFIG = {
    # Device
    'device': 'cuda',  # Will auto-fallback to 'cpu' if CUDA unavailable
    
    # Loss function
    'use_class_weights': True,  # Balance class imbalance
    'label_smoothing': 0.0,  # Label smoothing (0.0 = disabled)
    
    # Logging
    'log_interval': 10,  # Log every N batches
    'eval_interval': 1,  # Evaluate every N epochs
    
    # Reproducibility
    'seed': 42,
    'deterministic': True,
    
    # Mixed precision training (faster on modern GPUs)
    'use_amp': True,
}


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVAL_CONFIG = {
    'batch_size': 32,
    'metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'auc_roc',
        'confusion_matrix',
    ],
    
    # Visualization
    'save_confusion_matrix': True,
    'save_roc_curve': True,
    'save_predictions': True,
}


# ============================================================================
# DOMAIN ADAPTATION CONFIGURATION (For future use)
# ============================================================================

DOMAIN_ADAPT_CONFIG = {
    'source_domain': 'ct',
    'target_domain': 'xray',
    'method': 'dann',  # 'dann', 'coral', 'mmd', 'cyclegan'
    
    # DANN specific
    'dann_lambda': 1.0,
    'domain_classifier_hidden': 256,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_config(stage: str = 'stage1') -> dict:
    """
    Get configuration for specific training stage.
    
    Args:
        stage: 'stage1' (classifier only) or 'stage2' (fine-tuning)
    
    Returns:
        Combined configuration dictionary
    """
    stage_config = STAGE1_CONFIG if stage == 'stage1' else STAGE2_CONFIG
    
    config = {
        **MODEL_CONFIG,
        **stage_config,
        **DATA_CONFIG,
        **TRAIN_CONFIG,
        **EVAL_CONFIG,
    }
    
    # Create stage-specific checkpoint directory
    config['checkpoint_dir'].mkdir(exist_ok=True, parents=True)
    
    return config


def print_config(config: dict):
    """Pretty print configuration."""
    print("\n" + "=" * 70)
    print(f"Configuration: {config.get('name', 'Unknown')}")
    print("=" * 70)
    
    sections = [
        ('Model', ['architecture', 'num_classes', 'freeze_backbone']),
        ('Training', ['num_epochs', 'batch_size', 'learning_rate', 'optimizer']),
        ('Data', ['image_size', 'num_workers']),
        ('Device', ['device', 'use_amp']),
    ]
    
    for section_name, keys in sections:
        print(f"\n{section_name}:")
        for key in keys:
            if key in config:
                print(f"  {key}: {config[key]}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    """Test configuration."""
    
    print("Testing configuration files...")
    
    # Test Stage 1
    config1 = get_config('stage1')
    print_config(config1)
    
    # Test Stage 2
    config2 = get_config('stage2')
    print_config(config2)
    
    # Check paths
    print("\nPath Status:")
    print(f"  CT Data: {'✓' if CT_DATA_PATH.exists() else '✗'} {CT_DATA_PATH}")
    print(f"  X-ray Data: {'✓' if XRAY_DATA_PATH.exists() else '✗'} {XRAY_DATA_PATH}")
    print(f"  RadImageNet Weights: {'✓' if RADIMAGENET_WEIGHTS.exists() else '✗'} {RADIMAGENET_WEIGHTS}")
    print(f"  Checkpoints Dir: ✓ {CHECKPOINTS_DIR}")
    print(f"  Logs Dir: ✓ {LOGS_DIR}")
    print(f"  Results Dir: ✓ {RESULTS_DIR}")
    
    print("\n✓ Configuration loaded successfully!")
