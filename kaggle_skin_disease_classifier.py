"""
🏥 AI Skin Disease Classifier - Kaggle Notebook
================================================

This comprehensive notebook implements a CNN-based skin disease classifier
using transfer learning with EfficientNetB3.

📋 Dataset Structure Required:
/kaggle/input/skin-disease-dataset/
├── Acne/
├── Carcinoma/
├── Eczema/
├── Keratosis/
├── Milia/
└── Rosacea/

🎯 To use this notebook:
1. Upload your dataset to Kaggle Datasets with the structure above
2. Add the dataset to this notebook
3. Update DATASET_NAME below with your dataset name
4. Run all cells

Author: ML Lab
Version: 1.0.0
"""

# ============================================================================
# 📦 SECTION 1: INSTALLATION & IMPORTS
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB3, MobileNetV2, ResNet50V2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

# Scikit-learn for metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

print("="*70)
print("🏥 AI SKIN DISEASE CLASSIFIER")
print("="*70)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print("="*70)

# ============================================================================
# ⚙️ SECTION 2: CONFIGURATION
# ============================================================================

# 🔧 IMPORTANT: Update this with your Kaggle dataset name!
DATASET_NAME = 'skin-disease-dataset'  # Change this to your dataset name

# Paths
IS_KAGGLE = os.path.exists('/kaggle/input')
if IS_KAGGLE:
    DATA_DIR = f'/kaggle/input/{DATASET_NAME}'
    OUTPUT_DIR = '/kaggle/working'
else:
    DATA_DIR = './dataset'
    OUTPUT_DIR = './outputs'

# Create output directories
os.makedirs(f'{OUTPUT_DIR}/models', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/logs', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/outputs', exist_ok=True)

# Dataset configuration
CLASSES = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Milia', 'Rosacea']
NUM_CLASSES = len(CLASSES)

# Model configuration
MODEL_NAME = 'EfficientNetB3'
INPUT_SHAPE = (300, 300, 3)
WEIGHTS = 'imagenet'

# Training configuration - Phase 1
PHASE1_EPOCHS = 10
PHASE1_BATCH_SIZE = 32
PHASE1_LR = 1e-3

# Training configuration - Phase 2
PHASE2_EPOCHS = 40
PHASE2_BATCH_SIZE = 32
PHASE2_LR = 1e-5
UNFREEZE_RATIO = 0.3

# Data augmentation
VALIDATION_SPLIT = 0.15
ROTATION_RANGE = 20
WIDTH_SHIFT = 0.2
HEIGHT_SHIFT = 0.2
HORIZONTAL_FLIP = True
ZOOM_RANGE = 0.15
BRIGHTNESS_RANGE = [0.8, 1.2]

# Regularization
DROPOUT_1 = 0.4
DROPOUT_2 = 0.3
DENSE_UNITS = 256

# Callbacks
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Performance targets
TARGET_VAL_ACC = 0.85
TARGET_TEST_ACC = 0.82
TARGET_CARCINOMA_RECALL = 0.90

print("\n📋 Configuration:")
print(f"  Dataset: {DATA_DIR}")
print(f"  Model: {MODEL_NAME}")
print(f"  Input Shape: {INPUT_SHAPE}")
print(f"  Classes: {CLASSES}")
print(f"  Phase 1: {PHASE1_EPOCHS} epochs, LR={PHASE1_LR}")
print(f"  Phase 2: {PHASE2_EPOCHS} epochs, LR={PHASE2_LR}")

# ============================================================================
# 📊 SECTION 3: DATA EXPLORATION
# ============================================================================

print("\n" + "="*70)
print("📊 DATASET EXPLORATION")
print("="*70)

# Check if dataset exists
if not os.path.exists(DATA_DIR):
    print(f"❌ ERROR: Dataset not found at {DATA_DIR}")
    print(f"Please upload your dataset to Kaggle and update DATASET_NAME variable")
    sys.exit(1)

# Count images per class
print("\nDataset Statistics:")
total_images = 0
class_counts = {}

for class_name in CLASSES:
    class_path = os.path.join(DATA_DIR, class_name)
    if os.path.exists(class_path):
        count = len([f for f in os.listdir(class_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        class_counts[class_name] = count
        total_images += count
        print(f"  {class_name:12s}: {count:4d} images")
    else:
        print(f"  ⚠️ Warning: {class_name} directory not found")

print(f"\n  Total Images: {total_images}")
print(f"  Average per class: {total_images / len(class_counts):.1f}")

# Check balance
if len(set(class_counts.values())) == 1:
    print("  ✓ Dataset is perfectly balanced")
else:
    imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}")

# Visualize class distribution
plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values(), color='steelblue', alpha=0.8)
plt.title('Class Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Disease Class', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
for i, (cls, count) in enumerate(class_counts.items()):
    plt.text(i, count + 5, str(count), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/outputs/class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 🔄 SECTION 4: DATA PIPELINE
# ============================================================================

print("\n" + "="*70)
print("🔄 CREATING DATA PIPELINE")
print("="*70)

# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=ROTATION_RANGE,
    width_shift_range=WIDTH_SHIFT,
    height_shift_range=HEIGHT_SHIFT,
    horizontal_flip=HORIZONTAL_FLIP,
    zoom_range=ZOOM_RANGE,
    brightness_range=BRIGHTNESS_RANGE,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT
)

# Test data generator (normalization only)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=PHASE1_BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=PHASE1_BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

print(f"\n✓ Data pipeline created:")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")
print(f"  Class indices: {train_generator.class_indices}")

# Visualize augmented samples
print("\n📸 Visualizing data augmentation...")
sample_images, sample_labels = next(train_generator)

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.ravel()

for idx in range(9):
    axes[idx].imshow(sample_images[idx])
    class_idx = np.argmax(sample_labels[idx])
    axes[idx].set_title(CLASSES[class_idx], fontweight='bold')
    axes[idx].axis('off')

plt.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/outputs/augmentation_examples.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate class weights
y_train = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

print("\n⚖️ Class Weights (for handling imbalance):")
for idx, (class_name, weight) in enumerate(zip(CLASSES, class_weights)):
    print(f"  {class_name:12s}: {weight:.4f}")

# ============================================================================
# 🏗️ SECTION 5: MODEL ARCHITECTURE
# ============================================================================

print("\n" + "="*70)
print("🏗️ BUILDING MODEL ARCHITECTURE")
print("="*70)

def build_model(freeze_base=True):
    """Build CNN model with transfer learning"""

    # Load base model
    base_model = EfficientNetB3(
        include_top=False,
        weights=WEIGHTS,
        input_shape=INPUT_SHAPE
    )

    # Freeze or unfreeze base
    base_model.trainable = not freeze_base

    # Build custom head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(name='global_avg_pool'),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(DROPOUT_1, name='dropout_1'),
        layers.Dense(DENSE_UNITS, activation='relu', name='dense_1'),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(DROPOUT_2, name='dropout_2'),
        layers.Dense(NUM_CLASSES, activation='softmax', name='output')
    ], name=f'{MODEL_NAME}_SkinDiseaseClassifier')

    return model, base_model

# Build model
model, base_model = build_model(freeze_base=True)

print(f"\n✓ Model built: {MODEL_NAME}")
print(f"  Base Model Layers: {len(base_model.layers)}")
print(f"  Total Layers: {len(model.layers)}")

# Model summary
model.summary()

# Count parameters
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])

print(f"\n📊 Model Parameters:")
print(f"  Total: {trainable_params + non_trainable_params:,}")
print(f"  Trainable: {trainable_params:,}")
print(f"  Non-trainable: {non_trainable_params:,}")

# ============================================================================
# 🎓 SECTION 6: PHASE 1 TRAINING (Feature Extraction)
# ============================================================================

print("\n" + "="*70)
print("🎓 PHASE 1: FEATURE EXTRACTION")
print("="*70)
print(f"Epochs: {PHASE1_EPOCHS}")
print(f"Batch Size: {PHASE1_BATCH_SIZE}")
print(f"Learning Rate: {PHASE1_LR}")
print(f"Base Model: FROZEN")
print("="*70)

# Compile model
model.compile(
    optimizer=optimizers.Adam(learning_rate=PHASE1_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
phase1_callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LR,
        verbose=1
    ),
    ModelCheckpoint(
        f'{OUTPUT_DIR}/models/phase1_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train Phase 1
print("\n🚀 Starting Phase 1 training...")
history1 = model.fit(
    train_generator,
    epochs=PHASE1_EPOCHS,
    validation_data=val_generator,
    callbacks=phase1_callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# Save Phase 1 model
model.save(f'{OUTPUT_DIR}/models/phase1_final.h5')

print("\n✓ Phase 1 Complete!")
print(f"  Best Val Accuracy: {max(history1.history['val_accuracy']):.4f}")
print(f"  Final Train Accuracy: {history1.history['accuracy'][-1]:.4f}")

# Plot Phase 1 history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history1.history['accuracy'], label='Train', linewidth=2)
axes[0].plot(history1.history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_title('Phase 1: Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history1.history['loss'], label='Train', linewidth=2)
axes[1].plot(history1.history['val_loss'], label='Validation', linewidth=2)
axes[1].set_title('Phase 1: Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/outputs/phase1_history.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 🎓 SECTION 7: PHASE 2 TRAINING (Fine-Tuning)
# ============================================================================

print("\n" + "="*70)
print("🎓 PHASE 2: FINE-TUNING")
print("="*70)
print(f"Epochs: {PHASE2_EPOCHS}")
print(f"Batch Size: {PHASE2_BATCH_SIZE}")
print(f"Learning Rate: {PHASE2_LR}")
print(f"Unfreezing: Last {UNFREEZE_RATIO*100:.0f}% of layers")
print("="*70)

# Unfreeze last layers
total_layers = len(base_model.layers)
unfreeze_from = int(total_layers * (1 - UNFREEZE_RATIO))

print(f"\n🔓 Unfreezing layers {unfreeze_from} to {total_layers}...")

base_model.trainable = True
for layer in base_model.layers[:unfreeze_from]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=optimizers.Adam(learning_rate=PHASE2_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Count trainable parameters after unfreezing
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])

print(f"\n📊 Updated Parameters:")
print(f"  Trainable: {trainable_params:,}")
print(f"  Non-trainable: {non_trainable_params:,}")

# Callbacks for Phase 2
phase2_callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LR,
        verbose=1
    ),
    ModelCheckpoint(
        f'{OUTPUT_DIR}/models/phase2_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Recreate generators for Phase 2
train_generator_p2 = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=PHASE2_BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator_p2 = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=PHASE2_BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# Train Phase 2
print("\n🚀 Starting Phase 2 training...")
history2 = model.fit(
    train_generator_p2,
    epochs=PHASE2_EPOCHS,
    validation_data=val_generator_p2,
    callbacks=phase2_callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# Save final model
model.save(f'{OUTPUT_DIR}/models/final_model.h5')

print("\n✓ Phase 2 Complete!")
print(f"  Best Val Accuracy: {max(history2.history['val_accuracy']):.4f}")
print(f"  Final Train Accuracy: {history2.history['accuracy'][-1]:.4f}")

# Plot Phase 2 history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history2.history['accuracy'], label='Train', linewidth=2)
axes[0].plot(history2.history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_title('Phase 2: Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history2.history['loss'], label='Train', linewidth=2)
axes[1].plot(history2.history['val_loss'], label='Validation', linewidth=2)
axes[1].set_title('Phase 2: Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/outputs/phase2_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Combined history
combined_history = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss']
}

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(combined_history['accuracy'], label='Train', linewidth=2)
axes[0].plot(combined_history['val_accuracy'], label='Validation', linewidth=2)
axes[0].axvline(x=PHASE1_EPOCHS, color='r', linestyle='--', label='Phase 2 Start')
axes[0].set_title('Combined Training: Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(combined_history['loss'], label='Train', linewidth=2)
axes[1].plot(combined_history['val_loss'], label='Validation', linewidth=2)
axes[1].axvline(x=PHASE1_EPOCHS, color='r', linestyle='--', label='Phase 2 Start')
axes[1].set_title('Combined Training: Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/outputs/combined_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Save training history
with open(f'{OUTPUT_DIR}/outputs/training_history.pkl', 'wb') as f:
    pickle.dump(combined_history, f)

# ============================================================================
# 📈 SECTION 8: EVALUATION
# ============================================================================

print("\n" + "="*70)
print("📈 MODEL EVALUATION")
print("="*70)

# Load best model
best_model = keras.models.load_model(f'{OUTPUT_DIR}/models/phase2_best.h5')

# Reset validation generator
val_generator.reset()

# Get predictions
print("\n🔮 Generating predictions...")
predictions = best_model.predict(val_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

# Calculate accuracy
accuracy = np.mean(y_pred == y_true)
print(f"\n✓ Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification Report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
print(classification_report(y_true, y_pred, target_names=CLASSES))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Per-class metrics visualization
metrics_df = pd.DataFrame({
    'Class': CLASSES,
    'Precision': [report[cls]['precision'] for cls in CLASSES],
    'Recall': [report[cls]['recall'] for cls in CLASSES],
    'F1-Score': [report[cls]['f1-score'] for cls in CLASSES]
})

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(CLASSES))
width = 0.25

ax.bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
ax.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)

ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/outputs/per_class_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# Check critical metric: Carcinoma recall
carcinoma_recall = report['Carcinoma']['recall']

print("\n" + "="*70)
print("⚕️ CRITICAL METRIC CHECK - CARCINOMA RECALL")
print("="*70)
print(f"Current Recall: {carcinoma_recall:.4f}")
print(f"Target Recall: {TARGET_CARCINOMA_RECALL:.4f}")

if carcinoma_recall >= TARGET_CARCINOMA_RECALL:
    print("✓ TARGET MET - Excellent performance on cancer detection!")
else:
    print("✗ TARGET NOT MET - Consider model improvements")
print("="*70)

# Target achievement summary
print("\n" + "="*70)
print("🎯 TARGET ACHIEVEMENT SUMMARY")
print("="*70)

best_val_acc = max(combined_history['val_accuracy'])

targets = [
    ("Validation Accuracy", best_val_acc, TARGET_VAL_ACC),
    ("Test Accuracy", accuracy, TARGET_TEST_ACC),
    ("Carcinoma Recall", carcinoma_recall, TARGET_CARCINOMA_RECALL)
]

all_met = True
for name, achieved, target in targets:
    met = achieved >= target
    all_met = all_met and met
    status = "✓" if met else "✗"
    print(f"{status} {name:20s}: {achieved:.4f} >= {target:.4f}")

print("="*70)
if all_met:
    print("🎉 ALL TARGETS ACHIEVED! Model is ready for deployment.")
else:
    print("⚠️  Some targets not met. Consider further training.")
print("="*70)

# ============================================================================
# 💾 SECTION 9: SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("💾 SAVING RESULTS")
print("="*70)

# Save classification report
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(f'{OUTPUT_DIR}/outputs/classification_report.csv')
print(f"✓ Classification report saved")

# Save confusion matrix
cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
cm_df.to_csv(f'{OUTPUT_DIR}/outputs/confusion_matrix.csv')
print(f"✓ Confusion matrix saved")

# Save summary
summary = {
    'model': MODEL_NAME,
    'total_epochs': PHASE1_EPOCHS + PHASE2_EPOCHS,
    'best_val_accuracy': float(best_val_acc),
    'test_accuracy': float(accuracy),
    'carcinoma_recall': float(carcinoma_recall),
    'targets_met': all_met,
    'class_metrics': {cls: {
        'precision': float(report[cls]['precision']),
        'recall': float(report[cls]['recall']),
        'f1-score': float(report[cls]['f1-score'])
    } for cls in CLASSES}
}

import json
with open(f'{OUTPUT_DIR}/outputs/training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Training summary saved")

print("\n" + "="*70)
print("🎊 TRAINING COMPLETE!")
print("="*70)
print(f"\nModels saved in: {OUTPUT_DIR}/models/")
print(f"  - phase1_best.h5")
print(f"  - phase1_final.h5")
print(f"  - phase2_best.h5 (recommended)")
print(f"  - final_model.h5")
print(f"\nOutputs saved in: {OUTPUT_DIR}/outputs/")
print(f"  - Training plots")
print(f"  - Confusion matrix")
print(f"  - Classification report")
print(f"  - Training summary")
print("="*70)

print("\n✨ Next Steps:")
print("1. Download the models from /kaggle/working/models/")
print("2. Use phase2_best.h5 for deployment")
print("3. Test predictions on new images")
print("4. Deploy using FastAPI (see main repository)")
print("\n" + "="*70)
