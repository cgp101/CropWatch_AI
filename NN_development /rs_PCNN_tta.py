import torch
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
from pathlib import Path

from dataloaders import create_dataloaders
from SimplePestCNN_SE import SimplePestCNNWithAttention

def test_time_augmentation(model, test_loader, device):
    """
    Apply Test-Time Augmentation to improve predictions
    Averages predictions from multiple augmented versions
    """
    
    model.eval()
    
    # Define TTA transforms
    tta_transforms = [
        T.Compose([]),  # Original
        T.Compose([T.RandomHorizontalFlip(p=1.0)]),  # H-flip
        T.Compose([T.RandomVerticalFlip(p=1.0)]),  # V-flip  
        T.Compose([T.RandomRotation(degrees=10)]),  # Rotate 10
        T.Compose([T.RandomRotation(degrees=20)]),  # Rotate 20
    ]
    
    all_predictions = []
    all_labels = []
    
    print("Applying Test-Time Augmentation...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            batch_size = images.size(0)
            images = images.to(device)
            labels = labels.to(device)
            
            # Accumulate predictions from all augmentations
            batch_probs = torch.zeros(batch_size, 12).to(device)
            
            for transform in tta_transforms:
                # Apply transform
                if len(transform.transforms) > 0:
                    augmented_images = torch.stack([
                        transform(img) for img in images
                    ])
                else:
                    augmented_images = images
                
                # Get predictions
                outputs = model(augmented_images)
                probs = F.softmax(outputs, dim=1)
                batch_probs += probs
            
            # Average predictions
            batch_probs /= len(tta_transforms)
            _, predicted = torch.max(batch_probs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels)

def evaluate_with_tta(model_path):
    """Load model and evaluate with TTA"""
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model
    model = SimplePestCNNWithAttention(num_classes=12).to(device)
    model.load_state_dict(torch.load(model_path))
    
    # Load data
    data_dir = "/Users/cgp/Desktop/Portfolio/Crop_pest_identifier/pestven/Pest_data"
    _, _, test_loader, _ = create_dataloaders(
        data_dir, batch_size=32, num_workers=0, use_mps=True
    )
    
    # Standard evaluation
    print("\nStandard Evaluation (no TTA):")
    model.eval()
    standard_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            standard_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    standard_acc = accuracy_score(all_labels, standard_preds) * 100
    print(f"Standard Test Accuracy: {standard_acc:.2f}%")
    
    # TTA evaluation
    print("\nEvaluation with TTA:")
    tta_preds, tta_labels = test_time_augmentation(model, test_loader, device)
    
    # Calculate metrics
    tta_accuracy = accuracy_score(tta_labels, tta_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        tta_labels, tta_preds, average='weighted', zero_division=0
    )
    
    print(f"\nResults with TTA:")
    print(f"Test Accuracy: {tta_accuracy:.2f}")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")
    
    return tta_accuracy

if __name__ == "__main__":
    # Evaluate best attention model with TTA
    print("Evaluating SimplePestCNN with Attention + TTA")
    
    # Try on the original attention model
    model_path = "models/SimplePestCNN_attention.pth"
    if Path(model_path).exists():
        tta_acc = evaluate_with_tta(model_path)
    else:
        print(f"Model not found at {model_path}")
    
    # Also try on the best random search model if it exists
    best_rs_path = "models/SimplePestCNN_best_rs_se_SPCNN.pth"
    if Path(best_rs_path).exists():
        print(f"\nEvaluating Best SE block + Random Search Model + TTA")
        tta_acc_rs = evaluate_with_tta(best_rs_path)