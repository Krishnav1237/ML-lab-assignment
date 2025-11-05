# 🏥 AI Skin Disease Classifier - Multi-Dataset Training

A collection of 6 independent CNN training pipelines for different skin disease datasets using transfer learning with EfficientNetB3 on Kaggle.

## 📁 Repository Structure

This repository contains **6 separate dataset folders**, each with its own complete training pipeline:

```
ML-lab-assignment/
│
├── dataset1/           ✅ Contains: Training code (Skin Disease 6-class)
│   ├── kaggle_skin_disease_classifier.py
│   ├── config_kaggle.py
│   └── README.md
│
├── dataset2/           📁 Ready for your second dataset
│   └── README.md
│
├── dataset3/           📁 Ready for your third dataset
│   └── README.md
│
├── dataset4/           📁 Ready for your fourth dataset
│   └── README.md
│
├── dataset5/           📁 Ready for your fifth dataset
│   └── README.md
│
├── dataset6/           📁 Ready for your sixth dataset
│   └── README.md
│
├── KAGGLE_INSTRUCTIONS.md      Complete Kaggle guide
├── KAGGLE_QUICK_START.md       5-minute quick start
├── requirements.txt            Python dependencies
└── README.md                   This file
```

## 🎯 Purpose

Train **6 different models** on **6 different datasets** independently:
- Each dataset folder has its own complete pipeline
- Separate configurations per dataset
- Independent training notebooks
- Isolated outputs and models

## 📊 Current Status

| Dataset | Status | Description |
|---------|--------|-------------|
| **dataset1** | ✅ Ready | 6-class skin disease classifier (Acne, Carcinoma, Eczema, Keratosis, Milia, Rosacea) |
| **dataset2** | 📁 Empty | Awaiting configuration |
| **dataset3** | 📁 Empty | Awaiting configuration |
| **dataset4** | 📁 Empty | Awaiting configuration |
| **dataset5** | 📁 Empty | Awaiting configuration |
| **dataset6** | 📁 Empty | Awaiting configuration |

## 🚀 Quick Start - Dataset 1

### Step 1: Upload Dataset to Kaggle

Organize your images:
```
skin-disease-dataset/
├── Acne/
├── Carcinoma/
├── Eczema/
├── Keratosis/
├── Milia/
└── Rosacea/
```

Upload to https://www.kaggle.com/datasets

### Step 2: Create Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Create new notebook
3. Copy entire `dataset1/kaggle_skin_disease_classifier.py`
4. Paste into Kaggle notebook
5. Add your dataset (+ Add Data)
6. Enable GPU (Settings → GPU T4)
7. Enable Internet (Settings → Internet ON)

### Step 3: Run Training

1. Click "Run All"
2. Wait ~60 minutes
3. Download `phase2_best.h5` model

## 📖 Documentation

### General Guides
- [KAGGLE_INSTRUCTIONS.md](KAGGLE_INSTRUCTIONS.md) - Detailed Kaggle setup guide
- [KAGGLE_QUICK_START.md](KAGGLE_QUICK_START.md) - 5-minute quick reference

### Dataset-Specific
- [dataset1/README.md](dataset1/README.md) - Dataset 1 specific instructions
- Each dataset folder has its own README

## 🔧 How to Add New Datasets

To configure dataset2, dataset3, etc.:

1. **Copy files from dataset1**:
   ```bash
   cp dataset1/kaggle_skin_disease_classifier.py dataset2/
   cp dataset1/config_kaggle.py dataset2/
   ```

2. **Update configuration** in `dataset2/config_kaggle.py`:
   ```python
   DATASET_NAME = 'your-dataset2-name'
   CLASSES = ['Class1', 'Class2', ...]  # Your classes
   ```

3. **Update notebook** `dataset2/kaggle_skin_disease_classifier.py`:
   - Update `DATASET_NAME` variable
   - Update `CLASSES` list
   - Adjust hyperparameters if needed

4. **Follow same Kaggle workflow**:
   - Upload dataset to Kaggle
   - Create notebook
   - Run training

## 📦 What Each Dataset Folder Contains

Each dataset folder should have:

```
datasetN/
├── kaggle_skin_disease_classifier.py    Training notebook
├── config_kaggle.py                     Configuration
└── README.md                            Dataset-specific docs
```

After training on Kaggle, you'll download:
```
datasetN/
├── models/
│   ├── phase1_best.h5
│   ├── phase2_best.h5      ⭐ Use this
│   └── final_model.h5
└── outputs/
    ├── confusion_matrix.png
    ├── training_history.png
    └── summary.json
```

## 🎯 Training Specifications

### Common Settings (All Datasets)

- **Model**: EfficientNetB3 (pretrained on ImageNet)
- **Input Size**: 300×300×3
- **Training**: Two-phase (Feature Extraction + Fine-Tuning)
- **Phase 1**: 10 epochs, LR=1e-3, frozen base
- **Phase 2**: 40 epochs, LR=1e-5, unfrozen 30% layers
- **Batch Size**: 32
- **Augmentation**: Rotation, shift, zoom, brightness
- **Platform**: Kaggle with free GPU (T4/P100)

### Customizable per Dataset

- Number of classes
- Class names
- Epochs count
- Batch size
- Learning rates
- Augmentation parameters

## 💡 Use Cases

This multi-dataset structure is perfect for:

- **Multiple Projects**: Train different models for different clients/projects
- **A/B Testing**: Compare different datasets or preprocessing approaches
- **Ensemble Learning**: Train 6 models and combine predictions
- **Different Diseases**: Each dataset focuses on different conditions
- **Experimentation**: Try different architectures per dataset
- **Research**: Compare results across different data sources

## 📊 Expected Performance (per dataset)

| Metric | Target |
|--------|--------|
| Validation Accuracy | >85% |
| Test Accuracy | >82% |
| Training Time (GPU) | ~60 min |
| Training Time (CPU) | ~6-8 hours |

## 🔬 Example Workflow

### Scenario: Training All 6 Datasets

1. **Week 1**: Configure and train dataset1
2. **Week 2**: Configure and train dataset2
3. **Week 3**: Configure and train dataset3
4. **Week 4**: Configure and train dataset4
5. **Week 5**: Configure and train dataset5
6. **Week 6**: Configure and train dataset6

Or run all in parallel if you have multiple Kaggle accounts!

## ⚙️ Advanced Usage

### Batch Processing Multiple Datasets

```python
# After training all 6 models, ensemble predictions
models = []
for i in range(1, 7):
    model = load_model(f'dataset{i}/phase2_best.h5')
    models.append(model)

# Ensemble prediction
predictions = [model.predict(image) for model in models]
final_prediction = np.mean(predictions, axis=0)
```

### Comparing Model Performance

```python
# Load all 6 training summaries
summaries = []
for i in range(1, 7):
    with open(f'dataset{i}/outputs/summary.json') as f:
        summaries.append(json.load(f))

# Compare accuracies
for i, summary in enumerate(summaries, 1):
    print(f"Dataset {i}: {summary['test_accuracy']:.4f}")
```

## 📁 .gitignore

The following are ignored by git:
- `*/models/*.h5` (model files are large)
- `*/outputs/` (generated outputs)
- `*/logs/` (training logs)
- Dataset folders (add your data locally, not in git)

## 🤝 Contributing

Feel free to:
- Add more datasets (dataset7, dataset8, ...)
- Experiment with different architectures
- Share your configurations
- Report issues or improvements

## ⚠️ Medical Disclaimer

This is an AI tool for educational purposes. Always consult qualified medical professionals for diagnosis and treatment.

## 📧 Support

- **Issues**: Open a GitHub issue
- **Questions**: Check dataset-specific README files
- **Kaggle Help**: See KAGGLE_INSTRUCTIONS.md

---

## ✅ Next Steps

1. **Current**: Dataset1 is ready to use
2. **Your Task**: Configure datasets 2-6 based on your needs
3. **Process**: Copy dataset1 files, update configurations, train
4. **Result**: 6 independent trained models

---

**Built for scalable multi-dataset medical AI training on Kaggle**

*Powered by EfficientNetB3 | Free GPU Training | 6 Independent Pipelines*
