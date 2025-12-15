import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

import random
from pathlib import Path
import time

from dataloaders import create_dataloaders
from SimplePestCNN_SE import SimplePestCNNWithAttention


def mixup_data(x, y, alpha=1.0):
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_with_config(config, train_loader, val_loader, test_loader, device, epochs=30):
    """Train model with enhanced configuration including MixUp and Label Smoothing"""
    
    # Create model
    model = SimplePestCNNWithAttention(
        num_classes=12, 
        dropout_rate=config['dropout']
    ).to(device)
    
    # Loss with optional label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0))
    
    # Optimizer
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Scheduler
    if config['scheduler'] == 'OneCycleLR':
        scheduler = OneCycleLR(optimizer, max_lr=config['max_lr'], epochs=epochs, steps_per_epoch=len(train_loader))
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    mixup_alpha = config.get('mixup_alpha', 0)
    gradient_clip = config.get('gradient_clip', None)
    
    for epoch in range(epochs):
        # Training
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            # Apply MixUp
            if mixup_alpha > 0 and epoch > config.get('warmup_epochs', 0):
                images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if gradient_clip:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            if config['scheduler'] == 'OneCycleLR':
                scheduler.step()
        
        if config['scheduler'] == 'CosineAnnealing':
            scheduler.step()
        
        # Validation
        if epoch % 5 == 0 or epoch == epochs - 1:
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
            
            val_acc = 100. * val_correct / val_total
            best_val_acc = max(best_val_acc, val_acc)
    
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
    
    test_accuracy = accuracy_score(all_labels, all_preds) * 100
    
    return test_accuracy, best_val_acc, model


def random_search_enhanced(n_trials=5):
    """Enhanced random search with SE block hyperparameters"""
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = "/Users/cgp/Desktop/Portfolio/Crop_pest_identifier/pestven/Pest_data"
    
    # Optimized search space for SE-enhanced model
    search_space = {
        'lr': [0.0001, 0.0005, 0.001],
        'max_lr': [0.005, 0.01, 0.02],
        'dropout': [0.2, 0.3, 0.4], 
        'optimizer': ['Adam', 'AdamW'],  
        'scheduler': ['OneCycleLR', 'CosineAnnealing'],
        'weight_decay': [0, 1e-4, 5e-4],
        'batch_size': [16, 32],
        'label_smoothing': [0, 0.1], 
        'mixup_alpha': [0, 0.2],  
        'gradient_clip': [None, 5.0], 
        'warmup_epochs': [0, 2],  
    }
    
    best_accuracy = 0
    best_config = None
    best_model = None
    results = []
    
    print(f"Enhanced Random Search: {n_trials} trials")
    
    for trial in range(n_trials):
        config = {k: random.choice(v) for k, v in search_space.items()}
        
        print(f"\nTrial {trial+1}/{n_trials}")
        print(f"Config: LR={config['lr']}, Opt={config['optimizer']}, LS={config['label_smoothing']}, MixUp={config['mixup_alpha']}")
        
        train_loader, val_loader, test_loader, _ = create_dataloaders(
            data_dir, batch_size=config['batch_size'], num_workers=0, use_mps=True
        )
        
        start_time = time.time()
        test_acc, val_acc, model = train_with_config(
            config, train_loader, val_loader, test_loader, device, epochs=30
        )
        train_time = (time.time() - start_time) / 60
        
        print(f"Test: {test_acc:.1f}%, Val: {val_acc:.1f}%, Time: {train_time:.1f}min")
        
        results.append({
            'config': config,
            'test_accuracy': test_acc,
            'val_accuracy': val_acc,
            'time': train_time
        })
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_config = config
            best_model = model
            print(f"✓ New best: {test_acc:.2f}%")
    
    print(f"\n{'='*50}")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print("Best Config:")
    for k, v in best_config.items():
        if v is not None:
            print(f"  {k}: {v}")
    
    if best_model:
        Path("models").mkdir(exist_ok=True)
        torch.save(best_model.state_dict(), "models/SimplePestCNN_best_rs_se_SPCNN.pth")
        print(f"\nSaved: models/SimplePestCNN_best_rs_se_SPCNN.pth")
    
    return best_config, best_accuracy, results


if __name__ == "__main__":
    best_config, best_acc, all_results = random_search_enhanced(n_trials=5)
    
    print(f"\nSummary:")
    for i, res in enumerate(all_results, 1):
        print(f"Trial {i}: {res['test_accuracy']:.1f}% ({res['config']['optimizer']}, LS={res['config']['label_smoothing']}, MixUp={res['config']['mixup_alpha']})")