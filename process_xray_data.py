"""
Process X-ray dataset:
1. Remove PNEUMONIA folders
2. Downsample NORMAL to match COVID19 count (for balanced dataset)
3. Create train/val/test splits (70/15/15)
"""
import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

# Paths
source_root = "/home/sgram/.cache/kagglehub/datasets/prashant268/chest-xray-covid19-pneumonia/versions/2/Data"
target_root = "/home/sgram/.cache/kagglehub/datasets/prashant268/chest-xray-covid19-pneumonia/versions/2/Processed_XRay"

def count_files(directory):
    """Count image files in directory"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

def process_dataset():
    """Process the X-ray dataset"""
    
    print("=" * 70)
    print("Processing X-ray Dataset")
    print("=" * 70)
    
    # Collect all COVID19 and NORMAL images from train directory
    train_covid_dir = os.path.join(source_root, "train", "COVID19")
    train_normal_dir = os.path.join(source_root, "train", "NORMAL")
    
    # Get all image paths
    covid_images = [os.path.join(train_covid_dir, f) for f in os.listdir(train_covid_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    normal_images = [os.path.join(train_normal_dir, f) for f in os.listdir(train_normal_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nOriginal counts:")
    print(f"  COVID-19: {len(covid_images)}")
    print(f"  NORMAL: {len(normal_images)}")
    
    # Downsample NORMAL to match COVID19 count
    random.shuffle(normal_images)
    normal_images = normal_images[:len(covid_images)]
    
    print(f"\nAfter downsampling NORMAL:")
    print(f"  COVID-19: {len(covid_images)}")
    print(f"  NORMAL: {len(normal_images)}")
    
    # Combine and create splits
    all_images = {
        'COVID19': covid_images,
        'NORMAL': normal_images
    }
    
    # Create train/val/test splits (70/15/15)
    splits = {'train': 0.70, 'val': 0.15, 'test': 0.15}
    
    # Create target directory structure
    if os.path.exists(target_root):
        print(f"\nRemoving existing processed directory...")
        shutil.rmtree(target_root)
    
    for split in splits.keys():
        for class_name in all_images.keys():
            os.makedirs(os.path.join(target_root, split, class_name), exist_ok=True)
    
    # Split data for each class
    split_counts = defaultdict(lambda: defaultdict(int))
    
    for class_name, images in all_images.items():
        random.shuffle(images)
        total = len(images)
        
        train_count = int(total * splits['train'])
        val_count = int(total * splits['val'])
        
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]
        
        # Copy files to target directories
        for split_name, split_images in [('train', train_images), 
                                         ('val', val_images), 
                                         ('test', test_images)]:
            split_dir = os.path.join(target_root, split_name, class_name)
            
            for img_path in split_images:
                img_name = os.path.basename(img_path)
                target_path = os.path.join(split_dir, img_name)
                shutil.copy2(img_path, target_path)
                split_counts[split_name][class_name] += 1
    
    # Print split statistics
    print("\n" + "=" * 70)
    print("Split Statistics:")
    print("=" * 70)
    
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        total = 0
        for class_name in ['COVID19', 'NORMAL']:
            count = split_counts[split][class_name]
            print(f"  {class_name}: {count}")
            total += count
        print(f"  Total: {total}")
    
    print("\n" + "=" * 70)
    print(f"Processed dataset saved to: {target_root}")
    print("=" * 70)
    
    # Verify the splits
    print("\n" + "=" * 70)
    print("Verification:")
    print("=" * 70)
    for split in ['train', 'val', 'test']:
        print(f"\n{split}:")
        for class_name in ['COVID19', 'NORMAL']:
            class_dir = os.path.join(target_root, split, class_name)
            count = count_files(class_dir)
            print(f"  {class_name}: {count} files")

if __name__ == "__main__":
    process_dataset()
