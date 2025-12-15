# CropWatch-AI

**Canadian Agricultural Pest Detection & Risk Assessment System**

An end-to-end ML system combining computer vision for pest identification with a fine-tuned LLM for risk assessment, built for Canadian agricultural guidelines (CFIA, CCFM, GFO, Agriculture Canada).

---

## Project Overview

This system helps Canadian farmers identify crop pests and receive actionable risk assessments with economic thresholds, treatment recommendations, and loan risk scoring.

**Two main components:**
1. **Pest Classification** — EfficientNet ensemble (90.84% accuracy)
2. **Risk Assessment Agent** — Fine-tuned GPT-4.1-nano on Azure AI Foundry

---

## Features

- 12-class pest classification with ensemble inference
- Adaptive confidence thresholds via RLHF
- Dynamic ensemble weighting based on feedback
- LLM-powered pest management recommendations
- Agricultural loan risk scoring
- Real-time analytics dashboard

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT APP                            │
├─────────────┬─────────────────┬─────────────────────────────┤
│ Detection   │ Analytics       │ LLM Chat                    │
├─────────────┴─────────────────┴─────────────────────────────┤
│                 INFERENCE LAYER                             │
│    EfficientNet-B0 (224×224) + B4 (380×380) Ensemble       │
├─────────────────────────────────────────────────────────────┤
│                    RLHF LAYER                               │
│   Adaptive Thresholds + Dynamic Ensemble Weights            │
├─────────────────────────────────────────────────────────────┤
│                    LLM LAYER                                │
│        Fine-tuned GPT-4.1-nano (Azure AI Foundry)          │
├─────────────────────────────────────────────────────────────┤
│                  AZURE SERVICES                             │
│     Azure OpenAI  |  Azure Blob Storage  |  Canada Central │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset

**Crop Pests Dataset** from Kaggle: [https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset](https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset)

- **Training:** 11,502 images
- **Validation:** 1,095 images
- **Test:** 546 images
- **Classes:** 12 pest types (ants, bees, beetle, caterpillar, earthworms, earwig, grasshopper, moth, slug, snail, wasp, weevil)

---

## Model Development Journey

### Phase 1: Custom CNN from Scratch
Built a custom neural network to understand fundamentals before transfer learning:

- **SimplePestCNN** (424K params): 4 conv blocks, baseline architecture
- **PestCNN** (4.85M params): 5 conv blocks, deeper architecture
- **+ Squeeze-and-Excitation Blocks**: Channel attention mechanism
- **+ Class-weighted Loss**: sqrt dampening for imbalanced classes
- **+ Dynamic Augmentation**: Per-class augmentation factors (0.46x - 2.0x)

### Phase 2: Hyperparameter Optimization
Random search across 5 trials testing:
- Optimizers: Adam, AdamW, SGD
- Schedulers: OneCycleLR, CosineAnnealing
- Learning rates, dropout, weight decay

**Best config:** Adam, lr=0.0005, OneCycleLR, dropout=0.3, label_smoothing=0.1

### Phase 3: Transfer Learning & Ensemble
- EfficientNet-B0 (224×224 input): 90.11% accuracy
- EfficientNet-B4 (380×380 input): 89.74% accuracy
- Soft voting ensemble: **90.84% accuracy**

---

## Model Performance

### Pest Classification (12 classes)

| Phase | Model | Accuracy |
|-------|-------|----------|
| Baseline | SimplePestCNN | 50.92% |
| + SE Attention | Custom CNN | 60.26% |
| + Hyperparameter Tuning | Random Search | 65.38% |
| Transfer Learning | EfficientNet-B0 | 90.11% |
| Transfer Learning | EfficientNet-B4 | 89.74% |
| **Final Ensemble** | **B0 + B4 Soft Voting** | **90.84%** |

- **Top-5 Accuracy:** 94.14%
- **Inference Time:** ~80ms (ensemble), ~49ms (B0), ~199ms (B4)
- **Model Format:** ONNX Runtime optimized

### LLM Risk Assessment (CropWatch-AI)

| Metric | Value |
|--------|-------|
| Base Model | GPT-4.1-nano |
| Training Examples | 120 JSONL |
| Validation Examples | 60 JSONL |
| Token Accuracy | ~79% |
| Loss | 0.71 |
| Deployment | Azure AI Foundry (Canada Central) |

---

## RLHF System

The app implements Reinforcement Learning from Human Feedback with:

**Adaptive Thresholds:**
- Per-class confidence thresholds (50%-90% bounds)
- Graduated penalty system:
  - ✅ Correct: threshold × 0.95
  - ⚡ Top-5 match: threshold × 1.03
  - ❌ Wrong: threshold × 1.15

**Dynamic Ensemble Weighting:**
- Real-time accuracy tracking per model
- Automatic weight adjustment favoring better performer
- Weight history visualization

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML/DL | PyTorch, ONNX Runtime |
| Models | EfficientNet-B0/B4, GPT-4.1-nano |
| Cloud | Azure AI Foundry, Azure OpenAI, Azure Blob Storage |
| Frontend | Streamlit, Plotly |
| Data | JSONL, Pandas, NumPy |

---

## Project Structure

```
CropWatch_AI/
├── Connection_testers/
│   ├── openai_connection_tester.py
│   └── storage_connection_tester.py
├── Image_proces/
│   ├── img_aug.py
│   └── img_cleaner.py
├── NN_development/
│   ├── custom_built_nn.py
│   ├── dataloaders.py
│   ├── improving_simplePestCNN.py
│   ├── onnx_conversion.py
│   ├── rs_PCNN_tta.py
│   ├── rs+se_simplePCNN.py
│   ├── SimplePestCNN_SE.py
│   ├── train_custom_nn.py
│   └── transfer_learning_pretrain.py
├── system_prompt/
│   ├── canadian_agri_pest_training.jsonl
│   └── canadian_agri_pest_validation.jsonl
├── models/
│   ├── efficientnet_b0.onnx
│   └── efficientnet_b4.onnx
├── app.py                        # Main Streamlit application
├── app_health_check.py
├── analysis.py
├── data_import.py
├── conf_matrix.ipynb
├── croppest-classifier.ipynb
├── feedback_log.csv              # Auto-generated from RLHF
├── learned_thresholds.json       # Auto-generated from RLHF
├── requirements.txt
└── .env.example
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/cropwatch-ai.git
cd cropwatch-ai

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your Azure credentials

# Run health check
python utils/health_check.py

# Start the app
streamlit run app.py
```

---

## Environment Variables

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your_key
DEPLOYMENT_NAME=your_finetuned_model_name
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
```

---

## Knowledge Base Sources

- CFIA - Plant Health Evaluation Guidelines
- CCFM - Pest Risk Analysis Framework
- GFO - CropPests Guide
- Agriculture Canada - Climate Change Impacts

---

## Future Enhancements

Potential additions if extended:
- LangGraph workflow for multi-step reasoning
- Automated evaluation pipeline with drift detection
- Batch inference for large-scale farm assessments
- Mobile-friendly PWA version
- Integration with provincial crop insurance APIs

---

## License

MIT

---

## Acknowledgments

- Agriculture Canada
- Canadian Food Inspection Agency (CFIA)
- Canadian Council of Forest Ministers (CCFM)
- Grain Farmers of Ontario (GFO)