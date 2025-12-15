import torch
import torch.nn as nn
import torchvision.models as models
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import time
import json
from PIL import Image
import torchvision.transforms as transforms

# Import your model classes
from SimplePestCNN_SE import SimplePestCNNWithAttention

class EfficientNetWithDropout(nn.Module):
    """Match your training architecture"""
    def __init__(self, model_name='efficientnet_b0', num_classes=12, dropout_rate=0.2):
        super().__init__()
        if model_name == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(weights=None)
            num_features = self.base_model.classifier[1].in_features
        elif model_name == 'efficientnet_b4':
            self.base_model = models.efficientnet_b4(weights=None)
            num_features = self.base_model.classifier[1].in_features
        
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_features, num_classes)
        )
    def forward(self, x):
        return self.base_model(x)

def convert_to_onnx(model_name='ensemble'):
    """
    Convert PyTorch models to ONNX format
    """
    device = torch.device("cpu")
    
    print(f"Converting {model_name} to ONNX...")
    
    if model_name == 'custom':
        model = SimplePestCNNWithAttention(num_classes=12, dropout_rate=0.2).to(device)
        model.load_state_dict(torch.load("models/SimplePestCNN_best_rs_se_SPCNN.pth", map_location=device))
        dummy_input = torch.randn(1, 3, 224, 224)
        output_path = "models/pestcnn_custom.onnx"
        
    elif model_name == 'efficientnet_b0':
        model = EfficientNetWithDropout('efficientnet_b0', dropout_rate=0.2).to(device)
        model.load_state_dict(torch.load("models/efficientnet_b0_rs_configs.pth", map_location=device))
        dummy_input = torch.randn(1, 3, 224, 224)
        output_path = "models/efficientnet_b0.onnx"
        
    elif model_name == 'efficientnet_b4':
        model = EfficientNetWithDropout('efficientnet_b4', dropout_rate=0.2).to(device)
        model.load_state_dict(torch.load("models/efficientnet_b4_rs_configs.pth", map_location=device))
        dummy_input = torch.randn(1, 3, 380, 380)
        output_path = "models/efficientnet_b4.onnx"
        
    elif model_name == 'ensemble':
        print("Ensemble requires both B0 and B4 models...")
        convert_to_onnx('efficientnet_b0')
        convert_to_onnx('efficientnet_b4')
        return
    
    model.eval()
    
    # Export with simpler settings
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            
        )
    
    # Verify ONNX model
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"Model exported successfully to {output_path}")
        print(f"Size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"Warning during verification: {e}")
    
    return output_path

def test_onnx_model(onnx_path, test_image_path=None):
    """
    Test ONNX model with a sample image and compare with PyTorch
    """
    print(f"\nTesting ONNX model: {onnx_path}")
    
    # FIX: Convert to absolute path to handle external data files
    onnx_path = str(Path(onnx_path).absolute())
    
    # Class names for pest detection
    class_names = [
        'weevil', 'ants', 'bees', 'beetle', 'caterpillar', 
        'earthworms', 'earwig', 'grasshopper', 'moth', 
        'slug', 'snail', 'wasp'
    ]
    
    # Create ONNX Runtime session - simplified providers for Mac
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Get model input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_size = input_shape[2]  # Height/Width (assumes square input)
    
    print(f"Model expects input shape: {input_shape}")
    
    # Prepare test image
    if test_image_path and Path(test_image_path).exists():
        image = Image.open(test_image_path).convert('RGB')
    else:
        # Create random test image if none provided
        print("\nUsing random test image")
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).numpy()
    
    # Run inference
    start_time = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    inference_time = (time.time() - start_time) * 1000
    
    # Process outputs
    logits = outputs[0]
    probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=1)
    predicted_class = np.argmax(logits, axis=1)[0]
    confidence = probabilities[0][predicted_class].item()
    
    # Get top 5 predictions
    top5_probs, top5_indices = torch.topk(probabilities[0], 5)
    
    print(f"\n Inference Results:\n")
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"\nTop 5 predictions:\n")
    for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
        print(f"   {i+1}. {class_names[idx]}: {prob*100:.2f}%")
    
    return {
        'predicted_class': class_names[predicted_class],
        'confidence': float(confidence),
        'inference_time_ms': float(inference_time),
        'top5': [(class_names[idx.item()], float(prob)) for idx, prob in zip(top5_indices, top5_probs)]
    }


def test_ensemble_onnx(test_image_path=None):
    """
    Test ensemble of B0 and B4 ONNX models
    """
    print("\nTesting Ensemble ONNX models\n")
    
    # FIX: Use absolute paths
    b0_path = str(Path("models/efficientnet_b0.onnx").absolute())
    b4_path = str(Path("models/efficientnet_b4.onnx").absolute())
    
    if not Path(b0_path).exists() or not Path(b4_path).exists():
        print("Converting models first...")
        convert_to_onnx('ensemble')
    
    class_names = [
        'Weevil', 'ants', 'bees', 'beetle', 'catterpillar', 
        'earthworms', 'earwig', 'grasshopper', 'moth', 
        'slug', 'snail', 'wasp'
    ]
    
    # Load both models with simplified providers
    providers = ['CPUExecutionProvider']
    session_b0 = ort.InferenceSession(b0_path, providers=providers)
    session_b4 = ort.InferenceSession(b4_path, providers=providers)
    
    # Prepare test image
    if test_image_path and Path(test_image_path).exists():
        image = Image.open(test_image_path).convert('RGB')
        print(f"Using actual test image: {Path(test_image_path).name}")
    else:
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # Preprocess for B0 (224x224)
    transform_b0 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess for B4 (380x380)
    transform_b4 = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_b0 = transform_b0(image).unsqueeze(0).numpy()
    input_b4 = transform_b4(image).unsqueeze(0).numpy()
    
    # Run inference
    start_time = time.time()
    outputs_b0 = session_b0.run(None, {session_b0.get_inputs()[0].name: input_b0})
    outputs_b4 = session_b4.run(None, {session_b4.get_inputs()[0].name: input_b4})
    
    # Ensemble: average logits
    logits_b0 = torch.from_numpy(outputs_b0[0])
    logits_b4 = torch.from_numpy(outputs_b4[0])
    
    probs_b0 = torch.nn.functional.softmax(logits_b0, dim=1)
    probs_b4 = torch.nn.functional.softmax(logits_b4, dim=1)
    
    ensemble_probs = (probs_b0 + probs_b4) / 2
    inference_time = (time.time() - start_time) * 1000
    
    predicted_class = ensemble_probs.argmax(dim=1).item()
    confidence = ensemble_probs[0][predicted_class].item()
    
    print(f"\nEnsemble Inference Results:")
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence*100:.2f}%")
    
    return {
        'predicted_class': class_names[predicted_class],
        'confidence': float(confidence),
        'inference_time_ms': float(inference_time)
    }


if __name__ == "__main__":
    print("\nPEST DETECTION MODEL - ONNX CONVERSION & TESTING \n\n1. Converting models to ONNX format\n")
    
    # Convert models
    #convert_to_onnx('custom')
    #convert_to_onnx('efficientnet_b0')
    #convert_to_onnx('efficientnet_b4')
    
    # Test single model
    print("\n2. Testing individual ONNX models\n")
    
    #replace with actual pest image path)
    test_image = "pestven/Pest_data/test/images/ants-60-_jpg.rf.3c8f6f8d2824222706cf7007096e178a.jpg"
    results_b0 = test_onnx_model("models/efficientnet_b0.onnx", test_image)
    results_b4 = test_onnx_model("models/efficientnet_b4.onnx", test_image)
    
    # Test ensemble
    print("\n3. Testing ensemble\n")
    ensemble_results = test_ensemble_onnx(test_image)
    
    # Save results
    all_results = {
        'efficientnet_b0': results_b0,
        'efficientnet_b4': results_b4,
        'ensemble': ensemble_results
    }
    
    with open('models/onnx_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("ONNX conversion and testing complete!")
    print("Models ready for Azure deployment")
    print("Results saved to models/onnx_test_results.json")