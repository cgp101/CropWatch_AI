import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import time
from pathlib import Path

from dataloaders import create_dataloaders

class EfficientNetWithDropout(nn.Module):
    """EfficientNet with dropout=0.2 from Random Search config"""
    def __init__(self, model_name='efficientnet_b0', num_classes=12, dropout_rate=0.2):
        super().__init__()
        
        if model_name == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            num_features = self.base_model.classifier[1].in_features
        elif model_name == 'efficientnet_b4':
            self.base_model = models.efficientnet_b4(weights='IMAGENET1K_V1')
            num_features = self.base_model.classifier[1].in_features
        
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)

def train_with_best_config(model_name='efficientnet_b0'):
    """Train using Random search winning configuration"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    config = {
        'lr': 0.0005,
        'max_lr': 0.005,
        'dropout': 0.2,
        'batch_size': 8, # Reduced to 8 for MPS compatibility
        'label_smoothing': 0.1,
        'weight_decay': 0.0001,
        'epochs': 15
    }
    
    data_dir = "/Users/cgp/Desktop/Portfolio/Crop_pest_identifier/pestven/Pest_data"
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data_dir, batch_size=config['batch_size'], num_workers=0, use_mps=True
    )
    
    model = EfficientNetWithDropout(
        model_name=model_name,
        num_classes=12,
        dropout_rate=config['dropout']
    ).to(device)
    
    # Freeze backbone initially
    for param in model.base_model.features.parameters():
        param.requires_grad = False
    
    # optimizer: Adam
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    # OneCycleLR
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=config['max_lr'], 
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader)
    )
    
    # Early stopping setup
    best_val_acc = 0
    best_model_state = None
    patience = 3
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # Gradual unfreezing lets the model adapt smoothly 
        if epoch == 3:
            for param in model.base_model.features[-2:].parameters():
                param.requires_grad = True
        elif epoch == 5:
            for param in model.base_model.features.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(
                model.parameters(), 
                lr=config['lr'] * 0.1,
                weight_decay=config['weight_decay']
            )
            # Recreate scheduler with new optimizer
            scheduler = OneCycleLR(
                optimizer,
                max_lr=config['max_lr'] * 0.1,
                epochs=config['epochs'] - epoch,
                steps_per_epoch=len(train_loader)
            )
        
        # Training
        model.train()
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()  # Use scheduler throughout training
            
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train: {train_acc:.1f}%, Val: {val_acc:.1f}%")
        
        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Test with best model
    model.load_state_dict(best_model_state)
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(test_labels, test_preds) * 100
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='weighted', zero_division=0
    )
    
    train_time = (time.time() - start_time) / 60
    
    print(f"\n{model_name.upper()} Results:")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")
    print(f"Time: {train_time:.1f}min")
    
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"models/{model_name}_rs_configs.pth")
    
    return model, test_acc, precision, recall, f1

def ensemble_testing():
    """Test ensemble of B0 + B4"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model_b0 = EfficientNetWithDropout('efficientnet_b0', dropout_rate=0.2).to(device)
    model_b4 = EfficientNetWithDropout('efficientnet_b4', dropout_rate=0.2).to(device)
    
    model_b0.load_state_dict(torch.load("models/efficientnet_b0_rs_configs.pth"))
    model_b4.load_state_dict(torch.load("models/efficientnet_b4_rs_configs.pth"))
    
    model_b0.eval()
    model_b4.eval()
    
    data_dir = "/Users/cgp/Desktop/Portfolio/Crop_pest_identifier/pestven/Pest_data"
    _, _, test_loader, _ = create_dataloaders(data_dir, batch_size=32, num_workers=0, use_mps=True)
    
    ensemble_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Ensemble"):
            images, labels = images.to(device), labels.to(device)
            
            outputs_b0 = torch.softmax(model_b0(images), dim=1)
            outputs_b4 = torch.softmax(model_b4(images), dim=1)
            
            ensemble_output = (outputs_b0 + outputs_b4) / 2
            _, predicted = torch.max(ensemble_output, 1)
            
            ensemble_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    ensemble_acc = accuracy_score(all_labels, ensemble_preds) * 100
    
    # Calculate precision, recall, F1 for ensemble
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, ensemble_preds, average='weighted', zero_division=0
    )
    
    print(f"\nEnsemble Results:")
    print(f"Accuracy: {ensemble_acc:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")
    
    return ensemble_acc, precision, recall, f1

if __name__ == "__main__":
    #print("Training EfficientNet-B0\n")
    #model_b0, acc_b0, prec_b0, rec_b0, f1_b0 = train_with_best_config('efficientnet_b0')
    
    #print("\nTraining EfficientNet-B4\n")
    #model_b4, acc_b4, prec_b4, rec_b4, f1_b4 = train_with_best_config('efficientnet_b4')
    
    print("\nTesting Ensemble\n")
    ensemble_acc, ensemble_prec, ensemble_rec, ensemble_f1 = ensemble_testing()
    
    print("\nModel Comparision:\n")
    print(f"Custom CNN (from scratch): 69.41%")
    print(f"EfficientNet-B0: {acc_b0:.2f}% (Precision:{prec_b0*100:.1f}% Recall:{rec_b0*100:.1f}% F1 score:{f1_b0*100:.1f}%)")
    print(f"EfficientNet-B4: {acc_b4:.2f}% (Precision:{prec_b4*100:.1f}% Recall:{rec_b4*100:.1f}% F1 score:{f1_b4*100:.1f}%)")
    print(f"Ensemble: {ensemble_acc:.2f}% (Precision:{ensemble_prec*100:.1f}% Recall:{ensemble_rec*100:.1f}% F1 score:{ensemble_f1*100:.1f}%)")