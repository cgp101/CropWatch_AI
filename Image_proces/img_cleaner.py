import os
from pathlib import Path
from collections import Counter

def calculate_augmentation_needs(dataset_path, target_count=1050):
    """
    1. Merge WASP -> wasp by renaming files
    2. Calculate augmentation needs for classes with < 1050 images
    """
    dataset_path = Path(dataset_path)
    splits = ['train', 'valid', 'test']
    
    #print("\nMerging WASP -> wasp")
    
    # Rename WASP to wasp in all splits
    total_renamed = 0
    for split in splits:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if not images_dir.exists():
            continue
        
        # Find and rename WASP files
        wasp_images = list(images_dir.glob('WASP-*.jpg')) + list(images_dir.glob('WASP-*.jpeg'))
        
        for img_path in wasp_images:
            new_name = img_path.name.replace('WASP-', 'wasp-')
            new_img_path = img_path.parent / new_name
            img_path.rename(new_img_path)
            
            # Rename label file too
            label_name = img_path.name.replace('.jpg', '.txt').replace('.jpeg', '.txt')
            label_path = labels_dir / label_name
            
            if label_path.exists():
                new_label_name = new_name.replace('.jpg', '.txt').replace('.jpeg', '.txt')
                new_label_path = labels_dir / new_label_name
                label_path.rename(new_label_path)
            
            total_renamed += 1
        
        #print(f"\t{split}: Renamed {len(wasp_images)} WASP files to wasp")
    
    #print(f"\tTotal files renamed: {total_renamed}")
    
    # Now calculate augmentation needs
    train_images_dir = dataset_path / 'train' / 'images'
    image_files = list(train_images_dir.glob('*.jpg')) + list(train_images_dir.glob('*.jpeg'))
    
    # Count pest types
    pest_counts = Counter()
    for img in image_files:
        pest_name = img.stem.split('-')[0]
        pest_counts[pest_name] += 1
    
    print("\nCurrent Distribution & Augmentation Plan:")
    augmentation_needed = {}
    # Sort by count to show worst classes first
    for pest, count in sorted(pest_counts.items(), key=lambda x: x[1]):
        if count < target_count:
            needed = target_count - count
            factor = needed / target_count  # augmentation factor
            augmentation_needed[pest] = {
                'current': count,
                'needed': needed,
                'factor': factor
            }
            print(f"\t{pest:15s}: {count:4d} → {target_count} (need +{needed}) factor={factor:.3f}")
        else:
            print(f"\t{pest:15s}: {count:4d} (no change - already ≥ {target_count})")
    return augmentation_needed

if __name__ == '__main__':
    dataset_path = "/Users/cgp/Desktop/Portfolio/Crop_pest_identifier/pestven/Pest_data"
    augmentation_plan = calculate_augmentation_needs(dataset_path, target_count=1050)