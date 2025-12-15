import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import numpy as np
from collections import Counter

class PestDataset(Dataset):
    """
    PyTorch Dataset for pest classification
    Handles both classification and YOLO detection labels
    """
    
    def __init__(self, data_dir, split='train', transform=None, task='classification'):
        """
        Args:
            data_dir: Path to Pest_data folder
            split: 'train', 'valid', or 'test'
            transform: torchvision transforms to apply
            task: classification or detection
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.task = task
        self.transform = transform
        
        self.images_dir = self.data_dir / split / 'images'
        self.labels_dir = self.data_dir / split / 'labels'
        
        # Get all image paths
        self.image_paths = sorted(list(self.images_dir.glob('*.jpg')) + 
                                 list(self.images_dir.glob('*.jpeg')))
        
        # Create class mapping
        self.class_names = self._get_class_names()
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        print(f"\n{split.upper()} Dataset:")
        print(f"\tTotal images: {len(self.image_paths)}")
        print(f"\tNumber of classes: {len(self.class_names)}")
        
        if split == 'train':
            self._show_distribution()
    
    def _get_class_names(self):
        """Extract unique class names from filenames"""
        class_names = set()
        for img_path in self.image_paths:
            # Extract class name (everything before first hyphen)
            class_name = img_path.stem.split('-')[0]
            # Handle augmented images
            if 'aug' in img_path.stem:
                class_name = img_path.stem.split('-')[0]
            class_names.add(class_name)
        return sorted(list(class_names))
    
    def _show_distribution(self):
        """Show class distribution for training set"""
        class_counts = Counter()
        for img_path in self.image_paths:
            class_name = img_path.stem.split('-')[0]
            class_counts[class_name] += 1
        
        print("\tClass distribution:")
        for class_name in self.class_names:
            count = class_counts.get(class_name, 0)
            print(f"\t\t{class_name:15s}: {count:4d}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            For classification: (image, class_label)
            For detection: (image, boxes, labels)
        """
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get class label from filename
        class_name = img_path.stem.split('-')[0]
        class_label = self.class_to_idx[class_name]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.task == 'classification':
            return image, class_label
        
        elif self.task == 'detection':
            # Load YOLO labels
            label_path = self.labels_dir / (img_path.stem + '.txt')
            boxes = []
            labels = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert YOLO to xyxy format if needed
                            x1 = x_center - width/2
                            y1 = y_center - height/2
                            x2 = x_center + width/2
                            y2 = y_center + height/2
                            
                            boxes.append([x1, y1, x2, y2])
                            labels.append(class_id)
            
            boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
            labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
            
            return image, boxes, labels

def get_transforms(split, input_size=224):
    """
    Get appropriate transforms for each split
    M2 Mac friendly: moderate augmentation to save compute
    """
    if split == 'train':
        return T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                       std=[0.229, 0.224, 0.225])
        ])
    else:  # val and test
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

def create_dataloaders(data_dir, batch_size=32, num_workers=2, use_mps=True):
    """
    Create train/val/test dataloaders
    
    Args:
        batch_size: 32 for 8GB M2
        num_workers: 2 or 3 for M2 Mac
        use_mps: Enable Metal Performance Shaders
    """
    # Check device
    if use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n✓ Using Apple M2 GPU (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print(f"\n✓ Using CPU")
    
    # Create datasets
    train_dataset = PestDataset(data_dir, 'train', transform=get_transforms('train'))
    val_dataset = PestDataset(data_dir, 'valid', transform=get_transforms('valid'))
    test_dataset = PestDataset(data_dir, 'test', transform=get_transforms('test'))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=False  # Set False for M2 Mac
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"\nDataLoaders created:")
    print(f"\tTrain batches: {len(train_loader)} ({len(train_dataset)} images)")
    print(f"\tVal batches: {len(val_loader)} ({len(val_dataset)} images)")  
    print(f"\tTest batches: {len(test_loader)} ({len(test_dataset)} images)")
    print(f"\tBatch size: {batch_size}")
    print(f"\tDevice: {device}")
    
    return train_loader, val_loader, test_loader, device

if __name__ == '__main__':
    data_dir = "/Users/cgp/Desktop/Portfolio/Crop_pest_identifier/pestven/Pest_data"
    
    # Create dataloaders
    train_loader, val_loader, test_loader, device = create_dataloaders(
        data_dir,
        batch_size=32,  
        num_workers=2,
        use_mps=True
    )
    
    # Test loading a batch
    images, labels = next(iter(train_loader))
    print(f"\nTest batch loaded:")
    print(f"\tImages shape: {images.shape}")  # [32, 3, 224, 224]
    print(f"\tLabels shape: {labels.shape}")  # [32]
    print(f"\tMemory friendly for M2 Mac ✓")