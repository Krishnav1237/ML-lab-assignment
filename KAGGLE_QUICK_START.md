# 🚀 Kaggle Quick Start - 5 Minutes to Training

The fastest way to get your skin disease classifier running on Kaggle.

## ⚡ 3-Step Process

### Step 1: Upload Dataset (5 minutes)

1. Organize your images:
   ```
   skin-disease-dataset/
   ├── Acne/       (your images here)
   ├── Carcinoma/  (your images here)
   ├── Eczema/     (your images here)
   ├── Keratosis/  (your images here)
   ├── Milia/      (your images here)
   └── Rosacea/    (your images here)
   ```

2. Go to: https://www.kaggle.com/datasets
3. Click "New Dataset"
4. Upload the folder
5. Name it: **skin-disease-dataset**
6. Click "Create"

### Step 2: Create Notebook (2 minutes)

1. Go to: https://www.kaggle.com/code
2. Click "New Notebook"
3. Copy-paste **entire** `kaggle_skin_disease_classifier.py` file
4. Click "+ Add Data" → Add your dataset
5. Settings → Accelerator → **GPU T4**
6. Settings → Internet → **ON**

### Step 3: Run Training (1 click)

1. Click **"Run All"** button
2. Wait ~45-60 minutes
3. Download `phase2_best.h5` from Output tab

**That's it!** ✨

---

## 📥 Files to Download After Training

From the **Output** tab:

### Essential
- ✅ `models/phase2_best.h5` - **Your trained model**
- ✅ `outputs/training_summary.json` - **Performance metrics**

### Optional (for analysis)
- `outputs/confusion_matrix.png`
- `outputs/combined_history.png`
- `outputs/classification_report.csv`

---

## 🎯 Expected Results

After training completes, you should see:

```
🎯 TARGET ACHIEVEMENT SUMMARY
======================================================================
✓ Validation Accuracy  : 0.88+ >= 0.85
✓ Test Accuracy        : 0.85+ >= 0.82
✓ Carcinoma Recall     : 0.92+ >= 0.90
======================================================================
🎉 ALL TARGETS ACHIEVED! Model is ready for deployment.
```

---

## 🔧 Common Issues & Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| "Dataset not found" | Update `DATASET_NAME = 'your-dataset-name'` |
| "Out of memory" | Change `BATCH_SIZE = 16` (line ~80 & ~90) |
| Too slow | Enable GPU: Settings → GPU T4 |
| Can't download weights | Settings → Internet → ON |

---

## 📖 Need More Help?

- **Detailed Guide**: See [KAGGLE_INSTRUCTIONS.md](KAGGLE_INSTRUCTIONS.md)
- **Local Training**: See [README.md](README.md)
- **Issues**: Open a GitHub issue

---

## 🎓 What Happens During Training?

```
Minutes  | Phase                    | What's Happening
---------|--------------------------|----------------------------------
0-3      | Setup                    | Loading libraries, checking data
3-5      | Data Loading             | Creating augmented image pipeline
5-20     | Phase 1 (10 epochs)      | Training top layers only
20-60    | Phase 2 (40 epochs)      | Fine-tuning entire model
60-65    | Evaluation               | Testing, generating reports
```

**Total Time**: ~60 minutes with GPU (vs ~8 hours with CPU)

---

## ✅ Pre-Flight Checklist

Before clicking "Run All":

- [ ] Dataset uploaded to Kaggle
- [ ] Dataset named `skin-disease-dataset` (or updated in code)
- [ ] All 6 folders present (Acne, Carcinoma, Eczema, Keratosis, Milia, Rosacea)
- [ ] GPU enabled (check Settings → Accelerator)
- [ ] Internet ON (check Settings → Internet)
- [ ] Dataset added to notebook (check "+ Add Data")

**All checked?** Hit **Run All** and grab a coffee! ☕

---

## 🎉 After Training

### Test Your Model

```python
# Quick test in notebook (add new cell at end)
import numpy as np
from tensorflow.keras.preprocessing import image

# Load your trained model
model = keras.models.load_model('/kaggle/working/models/phase2_best.h5')

# Test on validation image
val_images, val_labels = next(val_generator)
test_img = val_images[0]
test_img = np.expand_dims(test_img, axis=0)

# Predict
pred = model.predict(test_img)
print(f"Predicted: {CLASSES[np.argmax(pred)]}")
print(f"Confidence: {np.max(pred):.2%}")
```

### Deploy with API

1. Download `phase2_best.h5`
2. Copy to local project: `models/best_model.h5`
3. Run API: `python app.py`
4. Access: http://localhost:8000

---

**You're ready to go! 🚀**

Just follow the 3 steps above and you'll have a production-ready skin disease classifier in about an hour!

Questions? Check [KAGGLE_INSTRUCTIONS.md](KAGGLE_INSTRUCTIONS.md) for detailed troubleshooting.
