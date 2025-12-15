import os
import shutil
from pathlib import Path

def load_data_splits(dataset_path):
    """Loads paths to train, validation and test data"""
    # Use the dataset_path parameter properly
    pest_data_dir = Path(dataset_path)  # Changed from Path('Pest_data')
    train_dir = pest_data_dir / 'train'
    val_dir = pest_data_dir / 'valid' 
    test_dir = pest_data_dir / 'test'
    
    if not all(d.exists() for d in [train_dir, val_dir, test_dir]):
        print(f"Error: Data directories not found at {pest_data_dir}")
        return None
        
    # Get lists of image files in each split
    train_files = list(train_dir.glob('**/*.jpg')) + list(train_dir.glob('**/*.jpeg'))
    val_files = list(val_dir.glob('**/*.jpg')) + list(val_dir.glob('**/*.jpeg'))
    test_files = list(test_dir.glob('**/*.jpg')) + list(test_dir.glob('**/*.jpeg'))
    
    print(f"Found {len(train_files)} training images")
    print(f"Found {len(val_files)} validation images")
    print(f"Found {len(test_files)} test images")
    
    return {
        'train': train_files,
        'valid': val_files,
        'test': test_files,
        'data_dir': pest_data_dir  # Added this line for dataset_analysis to work
    }

if __name__ == '__main__':
    data = load_data_splits(dataset_path="/Users/cgp/Desktop/Portfolio/Crop_pest_identifier/Pest_data")