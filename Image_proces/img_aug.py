import torch
import torchvision.transforms as T
from PIL import Image
import random
import numpy as np
from pathlib import Path
from collections import Counter
import shutil
from img_cleaner import calculate_augmentation_needs

class PestAugmentor:
    def __init__(self, dataset_path, target_count=1050):
        self.dataset_path = Path(dataset_path)
        self.target_count = target_count
        self.train_images_dir = self.dataset_path / 'train' / 'images'
        self.train_labels_dir = self.dataset_path / 'train' / 'labels'
        
    def get_transform_pipeline(self, factor):
        if factor > 0.2:
            return T.Compose([
                T.RandomRotation(degrees=30),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ])
        elif factor > 0.1:
            return T.Compose([
                T.RandomRotation(degrees=20),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.RandomAffine(degrees=0, scale=(0.9, 1.1))
            ])
        else:
            return T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ])
    
    def augment_image(self, image_path, transform):
        img = Image.open(image_path).convert('RGB')
        augmented_img = transform(img)
        return augmented_img
    
    def run_augmentation(self):
        print("\nStarting image augmentation...")
        plan = calculate_augmentation_needs(self.dataset_path, self.target_count)
        if not plan:
            print("No augmentation needed!")
            return
        print(f"\nClasses needing augmentation: {len(plan)}")
        total_created = 0
        for pest_class, info in plan.items():
            needed = info['needed']
            factor = info['factor']
            
            print(f"\n{pest_class}:")
            print(f"\tNeed {needed} new images (factor={factor:.3f})")
            
            class_images = []
            for ext in ['*.jpg', '*.jpeg']:
                class_images.extend(list(self.train_images_dir.glob(f"{pest_class}-{ext}")))
            
            if not class_images:
                print(f"\tNo images found for {pest_class}!")
                continue
            
            num_to_augment = min(needed, len(class_images))
            selected_images = random.sample(class_images, num_to_augment)
            
            transform = self.get_transform_pipeline(factor)
            
            created_count = 0
            for i, img_path in enumerate(selected_images[:needed]):
                try:
                    augmented_img = self.augment_image(img_path, transform)
                    
                    stem = img_path.stem
                    ext = img_path.suffix
                    new_name = f"{stem}-aug{i+1}{ext}"
                    new_path = self.train_images_dir / new_name
                    augmented_img.save(new_path)
                    label_name = img_path.stem + '.txt'
                    label_path = self.train_labels_dir / label_name
                    
                    if label_path.exists():
                        new_label_name = f"{stem}-aug{i+1}.txt"
                        new_label_path = self.train_labels_dir / new_label_name
                        shutil.copy2(label_path, new_label_path)
                    
                    created_count += 1
                    
                    if created_count % 50 == 0:
                        print(f"\t\tCreated {created_count}/{needed} images...")
                    
                except Exception as e:
                    print(f"\t\tError augmenting {img_path.name}: {e}")
                    continue
            
            print(f"\t Created {created_count} augmented images")
            total_created += created_count
        
        print(f"\nAugmentation complete! Total new images: {total_created}")
        self.verify_distribution()
    
    def verify_distribution(self):
        image_files = list(self.train_images_dir.glob('*.jpg')) + list(self.train_images_dir.glob('*.jpeg'))
        
        pest_counts = Counter()
        for img in image_files:
            stem = img.stem
            if '-aug' in stem:
                pest_name = stem.split('-')[0]
            else:
                pest_name = stem.split('-')[0]
            pest_counts[pest_name] += 1
        
        print("\nFinal distribution:")
        for pest, count in sorted(pest_counts.items()):
            status = "✓" if count >= self.target_count else "⚠"
            print(f"\t{pest:15s}: {count:4d} {status}")

if __name__ == '__main__':
    dataset_path = "/Users/cgp/Desktop/Portfolio/Crop_pest_identifier/pestven/Pest_data"
    augmentor = PestAugmentor(dataset_path, target_count=1050)
    augmentor.run_augmentation()