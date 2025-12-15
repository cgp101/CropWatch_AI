from pathlib import Path
from collections import Counter

def analyze_pest_dataset(dataset_path):
    """
    Analyzes pest dataset
    """
    dataset_path = Path(dataset_path)
    
    print("\nDataset Analysis Report for each split:\n")
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        images_dir = dataset_path / split / 'images'
        
        if not images_dir.exists():
            print(f"\n{split.upper()}: Directory not found")
            continue
            
        # Get all image files
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg'))
        pest_types = []
        for img in image_files:
            # Split filename: "ants-17-_.jpg" -> "ants"
            pest_name = img.stem.split('-')[0]
            pest_types.append(pest_name)
        
        # Count each pest type
        pest_counts = Counter(pest_types)
        
        # Display results
        print(f"\n{split.upper()} SET:")
        print(f"Total images: {len(image_files)}")
        print(f"Number of pest classes: {len(pest_counts)}")
        print("\nCount per pest type:")
        
        # Sort by count (descending)
        for pest, count in sorted(pest_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pest:20s}: {count:4d}")

if __name__ == '__main__':
    # Run analysis
    dataset_path = "/Users/cgp/Desktop/Portfolio/Crop_pest_identifier/pestven/Pest_data"
    analyze_pest_dataset(dataset_path)