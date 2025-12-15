import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import numpy as np
from pathlib import Path

from dataloaders import create_dataloaders
from custom_built_nn import SimplePestCNN

def calculate_class_weights():
    accuracies = torch.tensor([ 
        # Per-Class Accuracies for weight calculation
        0.879,  # Weevil
        0.870,  # ants
        0.780,  # bees
        0.150,  # beetle
        0.146,  # caterpillar
        0.593,  # earthworms
        0.119,  # earwig
        0.263,  # grasshopper
        0.578,  # moth
        0.239,  # slug
        0.568,  # snail
        0.870   # wasp
    ])
    
    # weights = 1.0 / (accuracies + 0.1) --> training 1 and 2 weights 
    # example weight calcution 
    # #0.150,  # beetle accuracy.            0.239,  # slug accuracy.
    # weight = 1.0 / (0.150 + 0.1) = 4.0.    weight = 1.0 / (0.239 + 0.1) = 3.33
    weights = 1.0 / torch.sqrt(accuracies + 0.1)
    # New weight calcution 
    #0.150,  # beetle accuracy.            0.239,  # slug accuracy.
    # weight = 1.0 / sqrt(0.150 + 0.1) = 2.0.    weight = 1.0 / sqrt(0.239 + 0.1) = 1.83
    weights = weights / weights.mean()
    return weights

def train_improved():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    data_dir = "/Users/cgp/Desktop/Portfolio/Crop_pest_identifier/pestven/Pest_data"
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data_dir, batch_size=32, num_workers=0, use_mps=True
    )
    
    model = SimplePestCNN(num_classes=12).to(device)
    
    # Class-weighted loss
    weights = calculate_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    
    best_val_acc = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(30):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/30'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Test
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")
    
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/SimplePestCNN_improved.pth")

if __name__ == "__main__":
    train_improved()
    import time
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    data_dir = "/Users/cgp/Desktop/Portfolio/Crop_pest_identifier/pestven/Pest_data"
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data_dir, batch_size=32, num_workers=0, use_mps=True
    )
    model = SimplePestCNN(num_classes=12).to(device)
model.load_state_dict(torch.load("models/SimplePestCNN_improved.pth"))
model.eval()

# Test for Top-5 accuracy
all_probs = []
all_labels = []
start_time = time.time()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate Top-5
top5_correct = 0
for i in range(len(all_labels)):
    top5_preds = np.argsort(all_probs[i])[-5:][::-1]
    if all_labels[i] in top5_preds:
        top5_correct += 1
top5_accuracy = (top5_correct / len(all_labels)) * 100

inference_time = time.time() - start_time

print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
print(f"Training  Time: {inference_time:.2f}s")