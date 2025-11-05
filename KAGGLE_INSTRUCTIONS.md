# 🏥 Kaggle Setup Guide - Skin Disease Classifier

Complete step-by-step guide to run this project on Kaggle.

## 📋 Table of Contents
1. [Preparing Your Dataset](#1-preparing-your-dataset)
2. [Uploading Dataset to Kaggle](#2-uploading-dataset-to-kaggle)
3. [Creating Kaggle Notebook](#3-creating-kaggle-notebook)
4. [Running the Training](#4-running-the-training)
5. [Downloading Results](#5-downloading-results)
6. [Troubleshooting](#troubleshooting)

---

## 1. Preparing Your Dataset

### Required Structure

Your dataset must be organized in this exact folder structure:

```
skin-disease-dataset/
├── Acne/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Carcinoma/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Eczema/
│   ├── image1.jpg
│   └── ...
├── Keratosis/
│   ├── image1.jpg
│   └── ...
├── Milia/
│   ├── image1.jpg
│   └── ...
└── Rosacea/
    ├── image1.jpg
    └── ...
```

### Requirements
- **6 folders**, one for each disease class
- **Recommended**: 399 images per class (2,394 total) for balanced dataset
- **Supported formats**: JPG, JPEG, PNG
- **Image quality**: Clear, well-lit skin condition images

---

## 2. Uploading Dataset to Kaggle

### Step-by-Step Upload Process

#### Option A: Using Kaggle Web Interface

1. **Go to Kaggle Datasets**
   - Navigate to: https://www.kaggle.com/datasets
   - Click **"New Dataset"** button

2. **Upload Your Dataset**
   - Click **"Upload"** or drag-and-drop your folder
   - Make sure you upload the parent folder with all 6 subfolders

3. **Configure Dataset**
   ```
   Title: Skin Disease Dataset
   Slug: skin-disease-dataset  ← IMPORTANT: Use this exact name!
   Description: Dataset containing 6 skin disease classes
   ```

4. **Set Visibility**
   - Choose **Private** (recommended for medical data)
   - Or **Public** if you want to share

5. **Click "Create"**
   - Wait for upload to complete (may take several minutes)

#### Option B: Using Kaggle API

```bash
# Install Kaggle API
pip install kaggle

# Configure API credentials
# Download kaggle.json from: https://www.kaggle.com/settings
# Place in ~/.kaggle/kaggle.json

# Create dataset metadata
cat > dataset-metadata.json << EOF
{
  "title": "Skin Disease Dataset",
  "id": "yourusername/skin-disease-dataset",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Create dataset
kaggle datasets create -p /path/to/skin-disease-dataset/

# Or update existing dataset
kaggle datasets version -p /path/to/skin-disease-dataset/ -m "Updated dataset"
```

### Verify Upload

After uploading, verify:
- All 6 folders are present
- Images are accessible
- Dataset is activated (may take a few minutes)

---

## 3. Creating Kaggle Notebook

### Method 1: Upload Pre-made Notebook (Recommended)

1. **Go to Kaggle Notebooks**
   - Navigate to: https://www.kaggle.com/code
   - Click **"New Notebook"**

2. **Upload Notebook File**
   - Click **"File"** → **"Upload Notebook"**
   - Select `kaggle_skin_disease_classifier.py` from this repository
   - Or copy-paste the entire content into a new notebook

3. **Add Dataset**
   - Click **"+ Add Data"** in the right sidebar
   - Search for your dataset: `skin-disease-dataset`
   - Click **"Add"**

4. **Enable GPU**
   - Click **"Settings"** in right sidebar
   - Under **"Accelerator"**, select **"GPU P100"** or **"GPU T4"**
   - Click **"Save"**

5. **Update Dataset Name**
   - Find this line in the notebook:
   ```python
   DATASET_NAME = 'skin-disease-dataset'  # Change if needed
   ```
   - Update if your dataset has a different name

### Method 2: Create from Scratch

If you prefer to type/paste manually:

1. Create new **Python** notebook (not R)
2. Copy all code from `kaggle_skin_disease_classifier.py`
3. Paste into notebook cells
4. Add your dataset
5. Enable GPU

---

## 4. Running the Training

### Before Starting

**Check Settings:**
- ✅ GPU is enabled (P100 or T4 recommended)
- ✅ Dataset is attached
- ✅ Internet is ON (for downloading pretrained weights)
- ✅ DATASET_NAME variable matches your dataset name

### Training Process

1. **Run All Cells**
   - Click **"Run All"** button at top
   - Or use keyboard shortcut: **Shift + Enter** through each cell

2. **Expected Timeline** (with GPU):
   ```
   Section 1-4: ~2-3 minutes (setup & data loading)
   Section 6: ~10-15 minutes (Phase 1 training)
   Section 7: ~30-40 minutes (Phase 2 training)
   Section 8-9: ~5 minutes (evaluation & saving)

   Total: ~45-60 minutes
   ```

3. **Monitor Progress**
   - Watch training progress bars
   - Check loss/accuracy after each epoch
   - Look for early stopping (if validation stops improving)

### What You'll See

```
📊 DATASET EXPLORATION
  ✓ Class distribution chart
  ✓ Sample images with augmentation

🎓 PHASE 1: FEATURE EXTRACTION
  ✓ 10 epochs of training
  ✓ Accuracy/loss plots

🎓 PHASE 2: FINE-TUNING
  ✓ 40 epochs of fine-tuning
  ✓ Combined training curves

📈 MODEL EVALUATION
  ✓ Confusion matrix
  ✓ Classification report
  ✓ Per-class metrics
  ✓ Target achievement summary
```

---

## 5. Downloading Results

### Files Created in `/kaggle/working/`

```
/kaggle/working/
├── models/
│   ├── phase1_best.h5        (Best model from Phase 1)
│   ├── phase1_final.h5       (Final Phase 1 model)
│   ├── phase2_best.h5        ⭐ USE THIS FOR DEPLOYMENT
│   └── final_model.h5        (Final Phase 2 model)
│
└── outputs/
    ├── class_distribution.png
    ├── augmentation_examples.png
    ├── phase1_history.png
    ├── phase2_history.png
    ├── combined_history.png
    ├── confusion_matrix.png
    ├── per_class_metrics.png
    ├── classification_report.csv
    ├── confusion_matrix.csv
    ├── training_history.pkl
    └── training_summary.json
```

### Download Methods

#### Method A: From Notebook Interface

1. Click **"Output"** tab at bottom of notebook
2. Expand `/kaggle/working/` folder
3. Right-click on files → **"Download"**
4. Download key files:
   - `models/phase2_best.h5` ⭐ (most important!)
   - All `.png` files for analysis
   - `.csv` and `.json` files for metrics

#### Method B: Create a Kaggle Dataset

1. At the end of notebook, add this cell:
```python
# This will save all outputs as a Kaggle dataset
!cp -r /kaggle/working/* /kaggle/output/
```

2. Click **"Save Version"** → **"Save & Run All"**
3. Go to **"Output"** tab when complete
4. Click **"Create Dataset"** on the output files

#### Method C: Programmatic Download

```python
# Add to end of notebook to download directly
from IPython.display import FileLink

# Download best model
display(FileLink('/kaggle/working/models/phase2_best.h5'))

# Download summary
display(FileLink('/kaggle/working/outputs/training_summary.json'))
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. "Dataset not found" Error

```python
❌ ERROR: Dataset not found at /kaggle/input/skin-disease-dataset
```

**Solutions:**
- Check dataset is attached: Click "+ Add Data" in sidebar
- Verify dataset name in code matches actual name
- Update `DATASET_NAME` variable:
  ```python
  DATASET_NAME = 'your-actual-dataset-name'
  ```

#### 2. "Out of Memory" Error

```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
- Reduce batch size:
  ```python
  PHASE1_BATCH_SIZE = 16  # Instead of 32
  PHASE2_BATCH_SIZE = 16  # Instead of 32
  ```
- Or use smaller model:
  ```python
  MODEL_NAME = 'MobileNetV2'
  INPUT_SHAPE = (224, 224, 3)
  ```

#### 3. Training is Too Slow

**If using CPU instead of GPU:**

- Enable GPU: Settings → Accelerator → GPU T4
- Verify GPU is working:
  ```python
  print(tf.config.list_physical_devices('GPU'))
  # Should show: [PhysicalDevice(name='/physical_device:GPU:0', ...)]
  ```

#### 4. Internet Off Error (Downloading Weights)

```
URLError: <urlopen error [Errno -3] Temporary failure in name resolution>
```

**Solutions:**
- Turn ON internet: Settings → Internet → ON
- Required for downloading ImageNet pretrained weights (one-time download)

#### 5. Session Timeout

Kaggle notebooks timeout after **9 hours** (12 hours with GPU).

**Solutions:**
- Save intermediate results:
  ```python
  model.save('/kaggle/working/checkpoint.h5')
  ```
- Resume from checkpoint if needed

#### 6. Validation Accuracy Not Improving

**If stuck at low accuracy:**

- Check dataset labels are correct
- Verify all images loaded properly
- Try increasing epochs:
  ```python
  PHASE2_EPOCHS = 60  # Instead of 40
  ```
- Check for data leakage or corruption

---

## 📊 Expected Performance

### Targets

With the default configuration and balanced dataset (399 images/class):

| Metric | Target | Expected |
|--------|--------|----------|
| **Validation Accuracy** | >85% | 85-92% |
| **Test Accuracy** | >82% | 82-90% |
| **Carcinoma Recall** | >90% | 90-95% |
| **Training Time (GPU)** | - | 45-60 min |

### Sample Output

```
🎯 TARGET ACHIEVEMENT SUMMARY
======================================================================
✓ Validation Accuracy  : 0.8876 >= 0.8500
✓ Test Accuracy        : 0.8542 >= 0.8200
✓ Carcinoma Recall     : 0.9250 >= 0.9000
======================================================================
🎉 ALL TARGETS ACHIEVED! Model is ready for deployment.
======================================================================
```

---

## 🚀 After Training

### 1. Download Best Model

```bash
# Download phase2_best.h5 from Kaggle
# This is your production-ready model
```

### 2. Test Predictions Locally

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model('phase2_best.h5')

# Prepare image
img = image.load_img('test_image.jpg', target_size=(300, 300))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
classes = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Milia', 'Rosacea']

print(f"Predicted: {classes[np.argmax(predictions)]}")
print(f"Confidence: {np.max(predictions):.2%}")
```

### 3. Deploy with FastAPI

Use the `app.py` from the main repository:

```bash
# Install dependencies
pip install fastapi uvicorn

# Copy your trained model
cp phase2_best.h5 models/best_model.h5

# Start API
python app.py

# Access at: http://localhost:8000
```

---

## 💡 Tips for Best Results

### 1. Data Quality
- Use high-quality, clear images
- Ensure consistent lighting
- Avoid blurry or low-resolution images
- Balance classes (same number of images per class)

### 2. Training Optimization
- Start with GPU enabled
- Monitor validation loss (stop if diverging)
- Use early stopping (already implemented)
- Save checkpoints frequently

### 3. Hyperparameter Tuning
```python
# Experiment with:
PHASE2_EPOCHS = 50  # More epochs if underfitting
DROPOUT_1 = 0.5     # Higher dropout if overfitting
PHASE2_LR = 5e-6    # Lower LR for fine-tuning
```

### 4. Model Selection
```python
# Try different architectures:
MODEL_NAME = 'MobileNetV2'     # Faster, lighter
MODEL_NAME = 'ResNet50V2'      # More robust
MODEL_NAME = 'EfficientNetB3'  # Best balance (default)
```

---

## 📚 Additional Resources

### Kaggle Documentation
- [Kaggle Notebooks Guide](https://www.kaggle.com/docs/notebooks)
- [Kaggle Datasets Guide](https://www.kaggle.com/docs/datasets)
- [Kaggle API Documentation](https://www.kaggle.com/docs/api)

### TensorFlow Resources
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

### This Project
- [Main README](README.md) - Complete project documentation
- [FastAPI Deployment](app.py) - API deployment code
- [Local Training](src/train.py) - Advanced training options

---

## 🤝 Support

### Getting Help

1. **Check Output Logs**: Read error messages carefully
2. **Verify Configuration**: Double-check all settings
3. **Search Kaggle Forums**: Many common issues solved there
4. **Open GitHub Issue**: Report bugs or ask questions

### Common Questions

**Q: Can I use this notebook for other datasets?**
A: Yes! Just organize your data in the same folder structure and update the `CLASSES` list.

**Q: How much GPU time does this use?**
A: ~1 hour with default settings. Kaggle provides 30 hours/week of GPU.

**Q: Can I resume training if interrupted?**
A: Yes, save model checkpoints and load them:
```python
model = keras.models.load_model('/kaggle/working/models/phase1_best.h5')
```

**Q: How do I improve accuracy?**
A: Try: more data, data cleaning, longer training, different augmentation, ensemble models.

---

## ✅ Quick Checklist

Before running, ensure:

- [ ] Dataset uploaded to Kaggle Datasets
- [ ] Dataset has correct folder structure (6 disease folders)
- [ ] Notebook created on Kaggle
- [ ] GPU is enabled (Settings → Accelerator → GPU)
- [ ] Internet is ON (Settings → Internet → ON)
- [ ] `DATASET_NAME` variable is correct in notebook
- [ ] Dataset is attached to notebook (+ Add Data)

After training:

- [ ] Check all targets are met (>85% val acc, >82% test acc, >90% Carcinoma recall)
- [ ] Download `phase2_best.h5` model
- [ ] Download visualization plots
- [ ] Download metrics CSV files
- [ ] Save Kaggle notebook (for future reference)

---

**Ready to train? Let's go! 🚀**

Upload your dataset, create the notebook, and run all cells. In ~1 hour, you'll have a production-ready skin disease classifier!

---

*Last updated: 2025*
*For issues or questions, please open a GitHub issue*
