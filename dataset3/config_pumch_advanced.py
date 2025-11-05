"""
Advanced Configuration for Dataset 3: PUMCH-ISD (8 Inflammatory Skin Diseases)

Dataset Details:
- Institution: Peking Union Medical College Hospital
- Total Images: ~9,748 (1,950 clinical + 7,798 dermoscopic)
- Patients: 1,174
- Modalities: Clinical photographs & Dermoscopic images
- Classes: 8 inflammatory skin diseases

Classes:
1. Psoriasis
2. Dermatitis (atopic, contact, seborrheic)
3. Lichen Planus
4. Pityriasis Rosea
5. Vitiligo
6. Acne Vulgaris
7. Rosacea
8. Morphea

This is the MOST advanced pipeline with multimodal support.
"""

import os

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# IMPORTANT: Update with your Kaggle dataset name!
DATASET_NAME = 'pumch-isd-inflammatory-skin-diseases'  # UPDATE THIS!

MODEL_ID = 'dataset3_pumch_isd'
MODEL_DESCRIPTION = 'Advanced 8-class inflammatory skin disease classifier (PUMCH-ISD)'

# Classes - 8 inflammatory diseases
CLASSES = [
    'Psoriasis',
    'Dermatitis',
    'Lichen_Planus',
    'Pityriasis_Rosea',
    'Vitiligo',
    'Acne_Vulgaris',
    'Rosacea',
    'Morphea'
]

NUM_CLASSES = len(CLASSES)

# Dataset modalities
HAS_CLINICAL_IMAGES = True
HAS_DERMOSCOPIC_IMAGES = True
MULTIMODAL = True  # If you want to train separate models for each modality

# ============================================================================
# KAGGLE PATHS
# ============================================================================

IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    DATA_DIR = f'/kaggle/input/{DATASET_NAME}'
    # For multimodal, you might have:
    # DATA_DIR_CLINICAL = f'/kaggle/input/{DATASET_NAME}/clinical'
    # DATA_DIR_DERMOSCOPIC = f'/kaggle/input/{DATASET_NAME}/dermoscopic'
    OUTPUT_DIR = '/kaggle/working'
else:
    DATA_DIR = './dataset_pumch_isd'
    OUTPUT_DIR = './outputs_pumch_isd'

MODEL_DIR = f'{OUTPUT_DIR}/models'
LOGS_DIR = f'{OUTPUT_DIR}/logs'
RESULTS_DIR = f'{OUTPUT_DIR}/results'
ENSEMBLE_DIR = f'{MODEL_DIR}/ensemble'

# ============================================================================
# MODEL ARCHITECTURE - ADVANCED FOR 8 CLASSES
# ============================================================================

MODEL_NAME = 'EfficientNetB5'  # Upgraded to B5 for 8-class problem
INPUT_SHAPE = (456, 456, 3)    # Even higher resolution for complex classification
WEIGHTS = 'imagenet'

# Robust architecture for multi-class
DENSE_UNITS = 768              # Larger for 8 classes
DROPOUT_1 = 0.5
DROPOUT_2 = 0.4
DROPOUT_3 = 0.3                # Additional dropout layer
USE_BATCH_NORM = True
USE_L2_REGULARIZATION = True
L2_LAMBDA = 0.0001

# Multi-head architecture option (for multimodal fusion)
USE_MULTI_HEAD = False         # Set True if combining clinical + dermoscopic
CLINICAL_HEAD_UNITS = 256
DERMOSCOPIC_HEAD_UNITS = 256

# ============================================================================
# TRAINING CONFIGURATION - PHASE 1: Feature Extraction (EXTENDED)
# ============================================================================

PHASE1_EPOCHS = 20             # Extended for 8 classes
PHASE1_BATCH_SIZE = 20         # Reduced for larger model
PHASE1_LR = 1e-3
PHASE1_FREEZE_BASE = True

# ============================================================================
# TRAINING CONFIGURATION - PHASE 2: Fine-Tuning (EXTENDED)
# ============================================================================

PHASE2_EPOCHS = 80             # Extended for complex 8-class problem
PHASE2_BATCH_SIZE = 20
PHASE2_LR = 3e-6               # Even lower LR for stability
UNFREEZE_RATIO = 0.5           # Unfreeze 50% for complex task

# ============================================================================
# ADVANCED DATA AUGMENTATION - MULTIMODAL
# ============================================================================

VALIDATION_SPLIT = 0.15

# Aggressive augmentation for large dataset
ROTATION_RANGE = 30
WIDTH_SHIFT = 0.3
HEIGHT_SHIFT = 0.3
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
ZOOM_RANGE = 0.25
BRIGHTNESS_RANGE = [0.6, 1.4]
SHEAR_RANGE = 0.2
CHANNEL_SHIFT_RANGE = 0.15

# Advanced augmentation
USE_CUTOUT = True
CUTOUT_SIZE = 50
USE_MIXUP = True
MIXUP_ALPHA = 0.3              # Slightly higher for multi-class
USE_CUTMIX = True              # Enable CutMix for diversity
CUTMIX_ALPHA = 1.0
USE_RANDAUGMENT = False        # Optional: RandAugment (very advanced)
RANDAUGMENT_N = 2
RANDAUGMENT_M = 10

# Test-Time Augmentation
USE_TTA = True
TTA_STEPS = 7                  # More steps for 8 classes

# ============================================================================
# PREPROCESSING - MULTIMODAL SPECIFIC
# ============================================================================

# Enhanced preprocessing for medical images
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_SIZE = (8, 8)

# Color normalization (important for dermoscopic images)
USE_COLOR_NORMALIZATION = True
COLOR_NORM_METHOD = 'macenko'  # Options: 'macenko', 'reinhard', 'vahadane'

# Image quality enhancement
USE_SHARPENING = True
SHARPEN_STRENGTH = 0.5

# Handle varying aspect ratios
ASPECT_RATIO_HANDLING = 'pad'  # pad with reflection
PADDING_MODE = 'reflect'

# ============================================================================
# CALLBACKS & OPTIMIZATION - ADVANCED
# ============================================================================

EARLY_STOPPING_PATIENCE = 15   # Longer patience for 8 classes
REDUCE_LR_PATIENCE = 6
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-9

# Advanced learning rate schedules
USE_COSINE_DECAY = True
USE_COSINE_RESTARTS = True     # Cosine annealing with warm restarts
RESTART_PERIOD = 20

USE_WARMUP = True
WARMUP_EPOCHS = 5

# Advanced optimization
USE_LABEL_SMOOTHING = True
LABEL_SMOOTHING = 0.15         # Higher for 8 classes
USE_CLASS_WEIGHTS = True       # Critical for imbalanced 8 classes

# Focal loss for hard examples
USE_FOCAL_LOSS = True          # Good for multi-class imbalance
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.5

# ============================================================================
# ENSEMBLE & CROSS-VALIDATION
# ============================================================================

# K-fold cross-validation (optional but recommended)
USE_KFOLD = False              # Set True for maximum performance
KFOLD_SPLITS = 5

# Snapshot ensemble
USE_SNAPSHOT_ENSEMBLE = True
SNAPSHOT_INTERVAL = 15         # Save every N epochs

# Bootstrap aggregating
USE_BAGGING = False            # Train multiple models with different seeds
N_BAGGING_MODELS = 3

# ============================================================================
# PERFORMANCE TARGETS - MEDICAL GRADE (8 CLASSES)
# ============================================================================

# Overall targets
TARGET_VAL_ACC = 0.88          # 8-class is harder than binary
TARGET_TEST_ACC = 0.85
TARGET_TOP2_ACC = 0.95         # Top-2 accuracy (important for similar diseases)
TARGET_MACRO_F1 = 0.86         # Balanced F1 across all classes

# Per-class minimum targets
MIN_CLASS_RECALL = 0.80        # Each class should have ≥80% recall
MIN_CLASS_PRECISION = 0.80     # Each class should have ≥80% precision

# Critical diseases (require higher performance)
CRITICAL_DISEASES = ['Morphea', 'Lichen_Planus']  # Rare/serious conditions
CRITICAL_DISEASE_RECALL = 0.85

# Confusion analysis
ACCEPTABLE_CONFUSION_PAIRS = [
    ('Dermatitis', 'Eczema'),  # Clinically similar
    ('Acne_Vulgaris', 'Rosacea'),  # Can be confused
    ('Psoriasis', 'Lichen_Planus')  # Similar appearance
]

# ============================================================================
# PER-CLASS MEDICAL IMPORTANCE
# ============================================================================

# Weight diseases by clinical importance
DISEASE_IMPORTANCE = {
    'Morphea': 1.5,            # Rare, needs careful detection
    'Lichen_Planus': 1.3,      # Important to catch
    'Psoriasis': 1.2,          # Chronic, needs management
    'Vitiligo': 1.2,           # Autoimmune, important
    'Dermatitis': 1.0,         # Common
    'Pityriasis_Rosea': 1.0,   # Self-limiting
    'Acne_Vulgaris': 1.0,      # Common
    'Rosacea': 1.0             # Common
}

# ============================================================================
# EVALUATION & EXPLAINABILITY
# ============================================================================

# Comprehensive evaluation
GENERATE_ROC_CURVE = True      # One-vs-rest for each class
GENERATE_PR_CURVE = True
GENERATE_CALIBRATION_PLOT = True
COMPUTE_CONFIDENCE_INTERVALS = True
BOOTSTRAP_ITERATIONS = 1000

# Per-class analysis
GENERATE_PER_CLASS_ROC = True
GENERATE_CONFUSION_HEATMAP = True
ANALYZE_MISCLASSIFICATIONS = True
TOP_K_ERRORS = 20              # Analyze top K misclassified images

# Explainability
USE_GRADCAM = True
GRADCAM_LAYER = 'auto'
SAVE_GRADCAM_SAMPLES = 30
GRADCAM_PER_CLASS = 5          # Sample N images per class

# Uncertainty quantification
USE_MC_DROPOUT = False         # Monte Carlo Dropout for uncertainty
MC_SAMPLES = 10

# ============================================================================
# MULTIMODAL FUSION (if using both clinical and dermoscopic)
# ============================================================================

# Fusion strategies
FUSION_STRATEGY = 'late'       # Options: 'early', 'late', 'intermediate'
FUSION_METHOD = 'average'      # Options: 'average', 'weighted', 'learned'

# Weights for weighted fusion (clinical, dermoscopic)
MODALITY_WEIGHTS = [0.4, 0.6]  # Dermoscopic might be more informative

# ============================================================================
# OUTPUT FILES
# ============================================================================

BEST_MODEL_PATH = f'{MODEL_DIR}/best_model_pumch.h5'
FINAL_MODEL_PATH = f'{MODEL_DIR}/final_model_pumch.h5'
PHASE1_MODEL_PATH = f'{MODEL_DIR}/phase1_model_pumch.h5'
PHASE2_MODEL_PATH = f'{MODEL_DIR}/phase2_model_pumch.h5'

# Ensemble models
ENSEMBLE_MODELS_PATHS = [
    f'{ENSEMBLE_DIR}/model_fold_{i}.h5' for i in range(1, 6)
]

TRAINING_HISTORY_PATH = f'{RESULTS_DIR}/training_history_pumch.pkl'
SUMMARY_JSON_PATH = f'{RESULTS_DIR}/summary_pumch.json'
METRICS_CSV_PATH = f'{RESULTS_DIR}/detailed_metrics_pumch.csv'
PER_CLASS_METRICS_PATH = f'{RESULTS_DIR}/per_class_metrics_pumch.csv'

# ============================================================================
# MEDICAL RECOMMENDATIONS (8 DISEASES)
# ============================================================================

RECOMMENDATIONS = {
    'Psoriasis': {
        'description': 'Chronic autoimmune condition with red, scaly plaques.',
        'severity': 'moderate-severe',
        'urgency': 'routine',
        'referral': 'Dermatologist',
        'treatment_options': [
            'Topical corticosteroids',
            'Vitamin D analogues',
            'Phototherapy for extensive cases',
            'Systemic therapy for severe cases',
            'Biologics for resistant cases'
        ],
        'monitoring': 'Regular follow-up every 3-6 months'
    },
    'Dermatitis': {
        'description': 'Inflammatory skin condition (atopic, contact, or seborrheic).',
        'severity': 'mild-moderate',
        'urgency': 'routine',
        'referral': 'Dermatologist if severe or persistent',
        'treatment_options': [
            'Identify and avoid triggers',
            'Topical corticosteroids',
            'Emollients and moisturizers',
            'Antihistamines for itching',
            'Consider patch testing for contact dermatitis'
        ],
        'monitoring': 'As needed based on symptoms'
    },
    'Lichen_Planus': {
        'description': 'Inflammatory condition affecting skin and mucous membranes.',
        'severity': 'moderate',
        'urgency': 'semi-urgent',
        'referral': 'Dermatologist recommended',
        'treatment_options': [
            'Topical corticosteroids',
            'Oral corticosteroids for severe cases',
            'Phototherapy',
            'Monitor oral lichen planus for malignant transformation',
            'Stress management'
        ],
        'monitoring': 'Regular monitoring, especially oral lesions'
    },
    'Pityriasis_Rosea': {
        'description': 'Self-limited viral rash with herald patch.',
        'severity': 'mild',
        'urgency': 'routine',
        'referral': 'Usually self-resolves',
        'treatment_options': [
            'Symptomatic treatment',
            'Antihistamines for itching',
            'Emollients',
            'Usually resolves in 6-8 weeks',
            'Reassurance and monitoring'
        ],
        'monitoring': 'Follow-up if persists >8 weeks'
    },
    'Vitiligo': {
        'description': 'Autoimmune destruction of melanocytes causing depigmentation.',
        'severity': 'moderate',
        'urgency': 'routine',
        'referral': 'Dermatologist for treatment options',
        'treatment_options': [
            'Topical corticosteroids',
            'Calcineurin inhibitors',
            'Phototherapy (PUVA, narrowband UVB)',
            'Surgical options for stable disease',
            'Camouflage makeup'
        ],
        'monitoring': 'Regular follow-up, check for other autoimmune conditions'
    },
    'Acne_Vulgaris': {
        'description': 'Common inflammatory condition of pilosebaceous units.',
        'severity': 'mild-moderate',
        'urgency': 'routine',
        'referral': 'Dermatologist if severe or scarring',
        'treatment_options': [
            'Topical retinoids',
            'Benzoyl peroxide',
            'Topical antibiotics',
            'Oral antibiotics for moderate-severe',
            'Isotretinoin for severe/resistant cases'
        ],
        'monitoring': 'Follow-up every 4-6 weeks during treatment'
    },
    'Rosacea': {
        'description': 'Chronic inflammatory condition with facial redness and telangiectasia.',
        'severity': 'mild-moderate',
        'urgency': 'routine',
        'referral': 'Dermatologist for persistent cases',
        'treatment_options': [
            'Identify and avoid triggers',
            'Topical metronidazole or azelaic acid',
            'Oral antibiotics (doxycycline)',
            'Laser therapy for telangiectasia',
            'Gentle skincare routine'
        ],
        'monitoring': 'Regular follow-up to adjust treatment'
    },
    'Morphea': {
        'description': 'Localized scleroderma with skin hardening and discoloration.',
        'severity': 'moderate-severe',
        'urgency': 'semi-urgent',
        'referral': 'Dermatologist and Rheumatologist',
        'treatment_options': [
            'Topical corticosteroids',
            'Topical calcineurin inhibitors',
            'Phototherapy',
            'Systemic therapy for extensive/progressive disease',
            'Physical therapy to prevent contractures'
        ],
        'monitoring': 'Close monitoring every 3 months, assess for systemic involvement'
    }
}

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration parameters"""
    assert NUM_CLASSES == 8, "Must have exactly 8 classes for PUMCH-ISD"
    assert PHASE1_EPOCHS > 0 and PHASE2_EPOCHS > 0, "Epochs must be positive"
    assert 0 < VALIDATION_SPLIT < 0.5, "Validation split must be between 0 and 0.5"
    assert INPUT_SHAPE[0] == INPUT_SHAPE[1], "Input must be square"
    assert len(CLASSES) == NUM_CLASSES, "Classes list must match NUM_CLASSES"
    assert all(disease in CLASSES for disease in CRITICAL_DISEASES), "Critical diseases must be in CLASSES"
    print("✓ Advanced configuration validated successfully")

# Print configuration summary
print("="*70)
print(f"ADVANCED CONFIGURATION: {MODEL_ID}")
print("="*70)
print(f"Dataset: PUMCH-ISD (Peking Union Medical College Hospital)")
print(f"Total Images: ~9,748 (1,950 clinical + 7,798 dermoscopic)")
print(f"Patients: 1,174")
print(f"Model: {MODEL_NAME}")
print(f"Input Shape: {INPUT_SHAPE}")
print(f"Classes: {NUM_CLASSES}")
for i, cls in enumerate(CLASSES, 1):
    importance = DISEASE_IMPORTANCE.get(cls, 1.0)
    print(f"  {i}. {cls:20s} (importance: {importance}x)")
print(f"\nTotal Epochs: {PHASE1_EPOCHS + PHASE2_EPOCHS}")
print(f"Multimodal: {'Yes' if MULTIMODAL else 'No'}")
print(f"\nAdvanced Features:")
print(f"  - EfficientNetB5: ✓")
print(f"  - Higher Resolution (456×456): ✓")
print(f"  - CLAHE + Color Norm: ✓")
print(f"  - Mixup + CutMix: {'✓' if USE_CUTMIX else 'Mixup only'}")
print(f"  - TTA ({TTA_STEPS} steps): {'✓' if USE_TTA else '✗'}")
print(f"  - Label Smoothing: {'✓' if USE_LABEL_SMOOTHING else '✗'}")
print(f"  - Focal Loss: {'✓' if USE_FOCAL_LOSS else '✗'}")
print(f"  - Snapshot Ensemble: {'✓' if USE_SNAPSHOT_ENSEMBLE else '✗'}")
print(f"  - K-Fold CV: {'✓' if USE_KFOLD else '✗'}")
print(f"  - Grad-CAM: {'✓' if USE_GRADCAM else '✗'}")
print(f"\nTarget Performance:")
print(f"  - Validation Accuracy: ≥{TARGET_VAL_ACC:.1%}")
print(f"  - Top-2 Accuracy: ≥{TARGET_TOP2_ACC:.1%}")
print(f"  - Macro F1: ≥{TARGET_MACRO_F1:.1%}")
print(f"  - Min Per-Class Recall: ≥{MIN_CLASS_RECALL:.1%}")
print("="*70)

# Validate on import
if __name__ != "__main__":
    validate_config()
