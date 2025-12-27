"""
Test-Time Augmentation (TTA) for higher accuracy.
Applies multiple augmentations at test time and averages predictions.
Can improve accuracy by 2-5%.
"""

import torch
from model import create_model
from dataloader import CTScanDataset, get_transforms
from config import CT_DATA_PATH
from tqdm import tqdm
from torchvision import transforms

def get_tta_transforms(img_size=224):
    """Get multiple augmented versions for TTA"""
    base_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Different augmentations
    tta_transforms = [
        base_transform,  # Original
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),  # Horizontal flip
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),  # Vertical flip
        transforms.Compose([
            transforms.Resize((int(img_size*1.1), int(img_size*1.1))),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),  # Center crop
        transforms.Compose([
            transforms.Resize((int(img_size*1.15), int(img_size*1.15))),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),  # Larger center crop
    ]
    
    return tta_transforms


def evaluate_with_tta(checkpoint_path, split='val'):
    """Evaluate model with Test-Time Augmentation"""
    
    # Load model
    model = create_model(
        pretrained_path='weights/RadImageNet-ResNet50.pth',
        num_classes=2,
        freeze_backbone=False,
        device='cuda'
    )
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset without transforms (we'll apply them manually)
    from PIL import Image
    import os
    
    data_root = str(CT_DATA_PATH)
    split_dir = os.path.join(data_root, split)
    
    samples = []
    for class_name in ['NONCOVID_CT', 'COVID_CT']:
        class_dir = os.path.join(split_dir, class_name)
        label = 0 if class_name == 'NONCOVID_CT' else 1
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append((os.path.join(class_dir, img_name), label))
    
    print(f"\nEvaluating on {split.upper()} set with TTA")
    print(f"Total samples: {len(samples)}")
    
    # Get TTA transforms
    tta_transforms = get_tta_transforms()
    print(f"Using {len(tta_transforms)} TTA augmentations")
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img_path, label in tqdm(samples, desc='TTA Evaluation'):
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Apply all TTA transforms and get predictions
            all_probs = []
            for transform in tta_transforms:
                img_tensor = transform(img).unsqueeze(0).cuda()
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                all_probs.append(probs)
            
            # Average predictions across all augmentations
            avg_probs = torch.stack(all_probs).mean(dim=0)
            pred = avg_probs.argmax(dim=1).item()
            
            if pred == label:
                correct += 1
            total += 1
    
    accuracy = 100. * correct / total
    print(f"\n{'='*70}")
    print(f"TTA RESULTS ({split.upper()} set):")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"{'='*70}")
    
    return accuracy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/stage1/best_model.pth')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    
    evaluate_with_tta(args.checkpoint, args.split)
