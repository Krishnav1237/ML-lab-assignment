"""
ROBUST Configuration for Dataset 2: Psoriasis & Lichen Planus Classification

Dataset Details:
- Total Images: ~1,500
- Training: ~1,200 images
- Testing: ~300 images
- Classes: Psoriasis, Lichen Planus (and related conditions)
- Format: JPEG RGB with varying resolutions
- Source: Dermnet

This configuration uses advanced techniques for medical-grade performance.
"""

import os

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# IMPORTANT: Update with your Kaggle dataset name!
DATASET_NAME = 'psoriasis-lichen-planus-skin-diseases'  # UPDATE THIS!

MODEL_ID = 'dataset2_psoriasis_lichen'
MODEL_DESCRIPTION = 'Robust Psoriasis & Lichen Planus classifier with advanced techniques'

# Classes - Update based on your actual dataset structure
# Option 1: Binary classification
CLASSES = ['Psoriasis', 'Lichen_Planus']

# Option 2: Multi-class (if you have more categories)
# CLASSES = ['Psoriasis', 'Lichen_Planus', 'Other_Dermatitis', 'Normal']

NUM_CLASSES = len(CLASSES)

# ============================================================================
# KAGGLE PATHS
# ============================================================================

IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    DATA_DIR = f'/kaggle/input/{DATASET_NAME}'
    OUTPUT_DIR = '/kaggle/working'
else:
    DATA_DIR = './dataset_psoriasis'
    OUTPUT_DIR = './outputs_psoriasis'

MODEL_DIR = f'{OUTPUT_DIR}/models'
LOGS_DIR = f'{OUTPUT_DIR}/logs'
RESULTS_DIR = f'{OUTPUT_DIR}/results'

# ============================================================================
# MODEL ARCHITECTURE - ROBUST CONFIGURATION
# ============================================================================

MODEL_NAME = 'EfficientNetB4'  # Upgraded to B4 for better performance
INPUT_SHAPE = (380, 380, 3)    # Higher resolution for better detail
WEIGHTS = 'imagenet'

# Advanced architecture
DENSE_UNITS = 512              # Increased capacity
DROPOUT_1 = 0.5                # Stronger regularization
DROPOUT_2 = 0.4
USE_BATCH_NORM = True
USE_L2_REGULARIZATION = True
L2_LAMBDA = 0.0001

# ============================================================================
# TRAINING CONFIGURATION - PHASE 1: Feature Extraction (EXTENDED)
# ============================================================================

PHASE1_EPOCHS = 15             # Extended from 10
PHASE1_BATCH_SIZE = 24         # Adjusted for larger input
PHASE1_LR = 1e-3
PHASE1_FREEZE_BASE = True

# ============================================================================
# TRAINING CONFIGURATION - PHASE 2: Fine-Tuning (EXTENDED)
# ============================================================================

PHASE2_EPOCHS = 60             # Extended from 40 for robust training
PHASE2_BATCH_SIZE = 24
PHASE2_LR = 5e-6               # Lower LR for stability
UNFREEZE_RATIO = 0.4           # Unfreeze more layers (40%)

# ============================================================================
# ADVANCED DATA AUGMENTATION - ROBUST PIPELINE
# ============================================================================

VALIDATION_SPLIT = 0.15

# Standard augmentation
ROTATION_RANGE = 25            # Increased rotation
WIDTH_SHIFT = 0.25             # Increased shift
HEIGHT_SHIFT = 0.25
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True           # Added vertical flip
ZOOM_RANGE = 0.2               # Increased zoom
BRIGHTNESS_RANGE = [0.7, 1.3]  # Wider brightness range
SHEAR_RANGE = 0.15             # Added shear transformation
CHANNEL_SHIFT_RANGE = 0.1      # Color channel shifts

# Advanced augmentation techniques
USE_CUTOUT = True              # Random erasing
CUTOUT_SIZE = 40
USE_MIXUP = True               # Mixup augmentation
MIXUP_ALPHA = 0.2
USE_CUTMIX = False             # CutMix (optional, can enable)
CUTMIX_ALPHA = 1.0

# Test-Time Augmentation (TTA)
USE_TTA = True                 # Predict with augmented versions
TTA_STEPS = 5                  # Number of augmented predictions to average

# ============================================================================
# PREPROCESSING FOR VARYING RESOLUTIONS
# ============================================================================

# Handle varying image sizes
RESIZE_METHOD = 'bilinear'     # or 'bicubic' for quality
ASPECT_RATIO_HANDLING = 'pad'  # Options: 'pad', 'crop', 'stretch'
PADDING_COLOR = (0, 0, 0)      # Black padding

# Image quality enhancement
USE_CLAHE = True               # Contrast Limited Adaptive Histogram Equalization
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

USE_DENOISING = False          # Optional: Image denoising
DENOISE_STRENGTH = 10

# ============================================================================
# CALLBACKS & OPTIMIZATION - ROBUST CONFIGURATION
# ============================================================================

EARLY_STOPPING_PATIENCE = 12   # Increased patience
REDUCE_LR_PATIENCE = 5         # Increased patience
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-8                  # Lower minimum

# Learning rate schedule
USE_COSINE_DECAY = True        # Cosine annealing
USE_WARMUP = True              # LR warmup
WARMUP_EPOCHS = 3

# Advanced optimization
USE_LABEL_SMOOTHING = True     # Prevent overconfidence
LABEL_SMOOTHING = 0.1
USE_CLASS_WEIGHTS = True       # Handle imbalance
USE_FOCAL_LOSS = False         # Alternative loss (optional)
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

# ============================================================================
# ENSEMBLE & CROSS-VALIDATION
# ============================================================================

USE_KFOLD = False              # K-fold cross-validation (time-intensive)
KFOLD_SPLITS = 5

USE_SNAPSHOT_ENSEMBLE = True   # Save multiple checkpoints
SNAPSHOT_INTERVAL = 10         # Every N epochs

# ============================================================================
# PERFORMANCE TARGETS - MEDICAL GRADE
# ============================================================================

TARGET_VAL_ACC = 0.90          # Higher target for medical use
TARGET_TEST_ACC = 0.88

# Medical-specific metrics
TARGET_SENSITIVITY = 0.92      # High sensitivity critical for disease detection
TARGET_SPECIFICITY = 0.88      # Good specificity to reduce false alarms
TARGET_AUC_ROC = 0.93          # Area under ROC curve
TARGET_F1_SCORE = 0.90         # Balanced performance

# Per-class targets (critical diseases need higher recall)
CLASS_SPECIFIC_TARGETS = {
    'Psoriasis': {'recall': 0.92, 'precision': 0.88},
    'Lichen_Planus': {'recall': 0.92, 'precision': 0.88}
}

# ============================================================================
# EVALUATION & EXPLAINABILITY
# ============================================================================

# Comprehensive evaluation
GENERATE_ROC_CURVE = True
GENERATE_PR_CURVE = True       # Precision-Recall curve
GENERATE_CALIBRATION_PLOT = True
COMPUTE_CONFIDENCE_INTERVALS = True
BOOTSTRAP_ITERATIONS = 1000

# Explainability
USE_GRADCAM = True
GRADCAM_LAYER = 'auto'         # Auto-detect last conv layer
SAVE_GRADCAM_SAMPLES = 20

# ============================================================================
# OUTPUT FILES
# ============================================================================

BEST_MODEL_PATH = f'{MODEL_DIR}/best_model_psoriasis.h5'
FINAL_MODEL_PATH = f'{MODEL_DIR}/final_model_psoriasis.h5'
PHASE1_MODEL_PATH = f'{MODEL_DIR}/phase1_model_psoriasis.h5'
PHASE2_MODEL_PATH = f'{MODEL_DIR}/phase2_model_psoriasis.h5'

# Ensemble models
ENSEMBLE_MODELS_DIR = f'{MODEL_DIR}/ensemble'

TRAINING_HISTORY_PATH = f'{RESULTS_DIR}/training_history_psoriasis.pkl'
SUMMARY_JSON_PATH = f'{RESULTS_DIR}/summary_psoriasis.json'
METRICS_CSV_PATH = f'{RESULTS_DIR}/detailed_metrics_psoriasis.csv'

# ============================================================================
# MEDICAL RECOMMENDATIONS (Optional - for deployment)
# ============================================================================

RECOMMENDATIONS = {
    'Psoriasis': {
        'description': 'Chronic autoimmune condition causing red, scaly patches on skin.',
        'severity': 'moderate',
        'next_steps': [
            'Consult dermatologist for treatment plan',
            'Consider topical corticosteroids or vitamin D analogs',
            'Phototherapy may be recommended for extensive cases',
            'Monitor for psoriatic arthritis symptoms',
            'Lifestyle modifications: stress management, moisturizing'
        ]
    },
    'Lichen_Planus': {
        'description': 'Inflammatory condition affecting skin and mucous membranes.',
        'severity': 'moderate',
        'next_steps': [
            'Dermatologist evaluation recommended',
            'Topical corticosteroids are first-line treatment',
            'Oral medications may be needed for severe cases',
            'Regular monitoring for oral lichen planus',
            'Avoid triggers: certain medications, stress'
        ]
    }
}

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration parameters"""
    assert NUM_CLASSES >= 2, "Must have at least 2 classes"
    assert PHASE1_EPOCHS > 0 and PHASE2_EPOCHS > 0, "Epochs must be positive"
    assert 0 < VALIDATION_SPLIT < 0.5, "Validation split must be between 0 and 0.5"
    assert INPUT_SHAPE[0] == INPUT_SHAPE[1], "Input must be square"
    assert DROPOUT_1 >= 0 and DROPOUT_1 < 1, "Dropout must be in [0, 1)"
    print("✓ Configuration validated successfully")

# Print configuration summary
print("="*70)
print(f"ROBUST CONFIGURATION: {MODEL_ID}")
print("="*70)
print(f"Dataset: {DATA_DIR}")
print(f"Model: {MODEL_NAME}")
print(f"Input Shape: {INPUT_SHAPE}")
print(f"Classes: {CLASSES}")
print(f"Total Epochs: {PHASE1_EPOCHS + PHASE2_EPOCHS}")
print(f"Advanced Features:")
print(f"  - Enhanced Augmentation: ✓")
print(f"  - Test-Time Augmentation: {'✓' if USE_TTA else '✗'}")
print(f"  - Mixup: {'✓' if USE_MIXUP else '✗'}")
print(f"  - CLAHE: {'✓' if USE_CLAHE else '✗'}")
print(f"  - Label Smoothing: {'✓' if USE_LABEL_SMOOTHING else '✗'}")
print(f"  - Snapshot Ensemble: {'✓' if USE_SNAPSHOT_ENSEMBLE else '✗'}")
print(f"  - Grad-CAM: {'✓' if USE_GRADCAM else '✗'}")
print(f"Target Performance:")
print(f"  - Validation Accuracy: ≥{TARGET_VAL_ACC:.1%}")
print(f"  - Sensitivity: ≥{TARGET_SENSITIVITY:.1%}")
print(f"  - AUC-ROC: ≥{TARGET_AUC_ROC:.1%}")
print("="*70)

# Validate on import
if __name__ != "__main__":
    validate_config()
