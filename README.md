# 🏥 AI Skin Disease Classifier

A production-ready deep learning system for multi-class classification of 6 skin conditions using CNN with transfer learning (EfficientNetB3), featuring Grad-CAM explainability and a FastAPI deployment.

## 🎯 Overview

This project implements a medical-grade skin disease classifier that can identify:
- **Acne** - Inflammatory skin condition
- **Carcinoma** - Skin cancer (critical detection)
- **Eczema** - Chronic inflammatory condition
- **Keratosis** - Benign skin growths
- **Milia** - Small white bumps
- **Rosacea** - Chronic facial redness

### Key Features

✅ **Transfer Learning** with EfficientNetB3 (300×300 input)
✅ **Two-Phase Training** (Feature Extraction → Fine-Tuning)
✅ **Data Augmentation** (rotation, shift, zoom, brightness)
✅ **Grad-CAM Explainability** (visual explanation of predictions)
✅ **Medical Recommendations** (care advice for each condition)
✅ **FastAPI Deployment** (RESTful API with Swagger UI)
✅ **Comprehensive Evaluation** (confusion matrix, per-class metrics)
✅ **High Recall for Cancer** (>90% target for Carcinoma detection)

## 📊 Performance Targets

- **Validation Accuracy**: >85%
- **Test Accuracy**: >82%
- **Carcinoma Recall**: >90% (critical for cancer detection)

## 🚀 Quick Start Options

### Option 1: Kaggle (Recommended for Training)

**Perfect for:** Training on cloud GPUs without local setup

1. **Upload dataset** to Kaggle Datasets
2. **Create notebook** using `kaggle_skin_disease_classifier.py`
3. **Enable GPU** and run all cells
4. **Download trained model** in ~1 hour

📖 **Full Guide**: [KAGGLE_INSTRUCTIONS.md](KAGGLE_INSTRUCTIONS.md)
⚡ **Quick Start**: [KAGGLE_QUICK_START.md](KAGGLE_QUICK_START.md)

### Option 2: Local Training

**Perfect for:** Custom workflows and development

#### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd ML-lab-assignment

# Install dependencies
pip install -r requirements.txt
```

#### 2. Prepare Dataset

Organize your dataset in the following structure:

```
dataset/
├── Acne/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Carcinoma/
│   ├── img1.jpg
│   └── ...
├── Eczema/
├── Keratosis/
├── Milia/
└── Rosacea/
```

Expected: 2,394 images total (399 per class)

#### 3. Train Model

```bash
# Train with default settings (EfficientNetB3, 2-phase training)
python src/train.py --data-dir dataset/

# Train with custom settings
python src/train.py \
    --data-dir dataset/ \
    --model EfficientNetB3 \
    --use-class-weights \
    --show-stats \
    --plot-history
```

**Training Phases:**
- **Phase 1** (10 epochs): Feature extraction with frozen base model
- **Phase 2** (40 epochs): Fine-tuning with unfrozen last 30% layers

#### 4. Evaluate Model

```bash
# Evaluate on test/validation set
python src/evaluate.py --model-path models/best_model.h5

# Evaluate with custom test directory
python src/evaluate.py \
    --model-path models/best_model.h5 \
    --test-dir test_dataset/ \
    --show-samples
```

#### 5. Make Predictions

```bash
# Predict single image
python src/predict.py path/to/image.jpg --visualize --gradcam

# Batch prediction (directory)
python src/predict.py path/to/images/ --batch

# With all options
python src/predict.py image.jpg \
    --model models/best_model.h5 \
    --show-probs \
    --visualize \
    --gradcam
```

#### 6. Deploy API

```bash
# Start FastAPI server
python app.py

# Or use uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Access Points:**
- **Homepage**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

**Test API:**
```bash
# Health check
curl http://localhost:8000/health

# Get classes
curl http://localhost:8000/classes

# Predict image
curl -X POST "http://localhost:8000/predict" \
     -F "file=@your_image.jpg"

# Get recommendations
curl http://localhost:8000/recommendation/Carcinoma
```

## 📁 Project Structure

```
ML-lab-assignment/
│
├── src/                          # Source code
│   ├── model.py                  # CNN architecture (EfficientNetB3)
│   ├── data_pipeline.py          # Data loading & augmentation
│   ├── train.py                  # Training script (2-phase)
│   ├── evaluate.py               # Evaluation with metrics
│   ├── predict.py                # Single/batch prediction
│   ├── gradcam.py                # Grad-CAM explainability
│   └── utils.py                  # Utility functions
│
├── app.py                        # FastAPI deployment
├── config.py                     # Configuration & hyperparameters
├── requirements.txt              # Python dependencies
│
├── models/                       # Saved models
│   ├── best_model.h5             # Best validation accuracy
│   ├── final_model.h5            # Final trained model
│   ├── phase1_model.h5           # After Phase 1
│   └── phase2_model.h5           # After Phase 2
│
├── outputs/                      # Training outputs
│   ├── training_history.pkl      # Training history
│   ├── confusion_matrix.png      # Confusion matrix
│   ├── classification_report.png # Per-class metrics
│   └── gradcam/                  # Grad-CAM visualizations
│
├── logs/                         # TensorBoard logs
│   ├── phase1/
│   └── phase2/
│
└── dataset/                      # Dataset (not included)
    ├── Acne/
    ├── Carcinoma/
    ├── Eczema/
    ├── Keratosis/
    ├── Milia/
    └── Rosacea/
```

## 🧠 Model Architecture

```python
EfficientNetB3 (pretrained on ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dropout(0.4)
    ↓
Dense(256, activation='relu')
    ↓
BatchNormalization
    ↓
Dropout(0.3)
    ↓
Dense(6, activation='softmax')  # 6 classes
```

### Hyperparameters

**Phase 1: Feature Extraction**
- Epochs: 10
- Batch Size: 32
- Learning Rate: 1e-3
- Base Model: Frozen

**Phase 2: Fine-Tuning**
- Epochs: 40
- Batch Size: 32
- Learning Rate: 1e-5
- Unfrozen: Last 30% of layers

**Regularization**
- Dropout: 0.4, 0.3
- Batch Normalization
- Early Stopping (patience=7)
- ReduceLROnPlateau (patience=3)

## 📊 Data Augmentation

- **Rotation**: ±20°
- **Width/Height Shift**: 20%
- **Horizontal Flip**: Yes
- **Zoom**: 15%
- **Brightness**: 80-120%

## 🔬 Grad-CAM Explainability

Generate visual explanations for model predictions:

```bash
# Generate Grad-CAM for single image
python src/gradcam.py --image path/to/image.jpg --model models/best_model.h5

# Via prediction script
python src/predict.py image.jpg --gradcam
```

Grad-CAM highlights the regions of the image that most influenced the model's decision, providing transparency and building trust in medical AI.

## 🌐 API Endpoints

### `POST /predict`
Upload image for disease prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@skin_image.jpg"
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "disease": "Acne",
    "confidence": 0.9234,
    "confidence_percentage": "92.34%",
    "low_confidence_warning": false
  },
  "all_probabilities": {
    "Acne": 0.9234,
    "Eczema": 0.0512,
    "Rosacea": 0.0134,
    ...
  },
  "recommendation": {
    "description": "Inflammatory skin condition...",
    "care_advice": [
      "Wash affected areas twice daily...",
      ...
    ],
    "urgency": "low"
  },
  "medical_disclaimer": "This is an AI-powered prediction..."
}
```

### `GET /health`
Health check endpoint

### `GET /classes`
Get supported disease classes

### `GET /recommendation/{disease}`
Get medical recommendations for specific disease

## 📈 Evaluation Metrics

The evaluation script generates:

1. **Confusion Matrix** - Visualize classification performance
2. **Classification Report** - Precision, Recall, F1-Score per class
3. **Per-Class Accuracy** - Individual class performance
4. **Carcinoma Recall Check** - Critical metric for cancer detection
5. **Training History** - Loss and accuracy curves

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model selection
MODEL_NAME = 'EfficientNetB3'  # or 'MobileNetV2', 'ResNet50V2'

# Training
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 40
PHASE1_LEARNING_RATE = 1e-3
PHASE2_LEARNING_RATE = 1e-5

# Data
VALIDATION_SPLIT = 0.15
BATCH_SIZE = 32

# Paths
DATA_DIR = './dataset'
MODEL_DIR = './models'
```

## 🛠️ Development

### Run Tests

```bash
# Test model architecture
python src/model.py

# Test data pipeline
python src/data_pipeline.py

# Test Grad-CAM
python src/gradcam.py --image test_image.jpg
```

### Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir logs/
```

Access at: http://localhost:6006

## 📝 Requirements

### Core Dependencies
- TensorFlow >= 2.13.0
- Keras >= 2.13.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0

### Visualization
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0

### Deployment
- FastAPI >= 0.100.0
- Uvicorn >= 0.23.0

### Full list in `requirements.txt`

## 🎯 Performance Tips

1. **Use GPU** for faster training (30-60 mins on T4 GPU vs hours on CPU)
2. **Increase epochs** if not converging (try 60-80 for Phase 2)
3. **Adjust learning rate** if loss plateaus too early
4. **Use class weights** if dataset becomes imbalanced
5. **Monitor validation accuracy** to prevent overfitting

## ⚠️ Medical Disclaimer

**IMPORTANT**: This is an AI-powered prediction tool and should **NOT** replace professional medical diagnosis. Always consult a qualified dermatologist or healthcare provider for proper diagnosis and treatment.

The model achieves high accuracy on the training dataset but may not generalize to all real-world scenarios. Use as a supplementary tool only.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Ensure compliance with medical AI regulations in your jurisdiction before deployment.

## 🙏 Acknowledgments

- EfficientNet paper: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- Grad-CAM paper: [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)
- Transfer learning resources from TensorFlow/Keras documentation

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ for medical AI research**
