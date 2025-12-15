import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from tqdm import tqdm
import time
import json
from pathlib import Path
import numpy as np

from dataloaders import create_dataloaders
from custom_built_nn import PestCNN, SimplePestCNN

class ModelTrainer:
    def __init__(self, model, model_name, device, num_classes=12):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.num_classes = num_classes
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0
        self.best_model_state = None
    
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Training')
        
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        return running_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return running_loss / len(val_loader), 100. * correct / total
    
    def train(self, train_loader, val_loader, epochs=20):
        print(f"\nTraining {self.model_name}")
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if epoch > 10 and val_acc < self.history['val_acc'][-5]:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        self.history['training_time'] = training_time
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"Training completed in {training_time/60:.2f} minutes")
        return self.history
    
    def compute_metrics(self, test_loader):
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = accuracy_score(all_labels, all_predictions) * 100
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        ) 
        top5_correct = 0
        for i in range(len(all_labels)):
            top5_preds = np.argsort(all_probs[i])[-5:][::-1]
            if all_labels[i] in top5_preds:
                top5_correct += 1
        top5_accuracy = (top5_correct / len(all_labels)) * 100
        
        per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'top5_accuracy': top5_accuracy,
            'per_class_metrics': {
                'precision': per_class_precision.tolist(),
                'recall': per_class_recall.tolist(),
                'f1': per_class_f1.tolist()
            }
        }
        return metrics

def train_custom_model(model, model_name, train_loader, val_loader, test_loader, device, epochs=25):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{model_name}: {trainable_params:,} parameters")
    
    trainer = ModelTrainer(model, model_name, device)
    history = trainer.train(train_loader, val_loader, epochs=epochs)
    
    print("\nComputing test metrics...")
    metrics = trainer.compute_metrics(test_loader)
    
    results = {
        'model_name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'metrics': metrics,
        'best_val_acc': trainer.best_val_acc,
        'training_time': history['training_time'],
        'history': history
    }
    
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"models/{model_name}_best.pth")
    
    print(f"\n{model_name} Results:")
    print(f"\tAccuracy: {metrics['accuracy']:.2f}%")
    print(f"\tPrecision: {metrics['precision']:.2f}%")
    print(f"\tRecall: {metrics['recall']:.2f}%")
    print(f"\tF1-Score: {metrics['f1_score']:.2f}%")
    print(f"\tTop-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
    print(f"\tTraining Time: {history['training_time']/60:.2f} minutes")
    
    return results

def main():
    data_dir = "/Users/cgp/Desktop/Portfolio/Crop_pest_identifier/pestven/Pest_data"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nLoading data...")
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data_dir, batch_size=32, num_workers=2, use_mps=True
    )
    
    print(f"\nTRAINING CUSTOM MODELS")
    
    # Train SimplePestCNN
    simple_model = SimplePestCNN(num_classes=12).to(device)
    simple_results = train_custom_model(
        simple_model, "SimplePestCNN", 
        train_loader, val_loader, test_loader, 
        device, epochs=25
    )
    
    if device.type == 'mps':
        torch.mps.empty_cache()
    
    # Train PestCNN
    full_model = PestCNN(num_classes=12).to(device)
    full_results = train_custom_model(
        full_model, "PestCNN", 
        train_loader, val_loader, test_loader,
        device, epochs=25
    )
    
    print(f"\nCUSTOM MODELS COMPARISON")
    print(f"{'Metric':<20} {'SimplePestCNN':<15} {'PestCNN':<15}")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'top5_accuracy']:
        simple_val = simple_results['metrics'][metric]
        full_val = full_results['metrics'][metric]
        print(f"{metric:<20} {simple_val:<15.2f} {full_val:<15.2f}")
    
    print(f"{'Training Time (min)':<20} {simple_results['training_time']/60:<15.2f} {full_results['training_time']/60:<15.2f}")
    print(f"{'Parameters':<20} {simple_results['trainable_params']:<15,} {full_results['trainable_params']:<15,}")
    
    results = {
        'SimplePestCNN': simple_results,
        'PestCNN': full_results
    }
    
    with open("models/custom_models_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Models saved to 'models/' directory")
    print(f"✓ Results saved to 'models/custom_models_results.json'")

if __name__ == "__main__":
    main()