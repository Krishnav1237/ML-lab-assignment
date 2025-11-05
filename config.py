"""
Configuration file for Skin Disease Classifier
"""
import os

# Dataset configuration
CLASSES = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Milia', 'Rosacea']
NUM_CLASSES = len(CLASSES)
TOTAL_IMAGES = 2394
IMAGES_PER_CLASS = 399

# Model configuration
MODEL_NAME = 'EfficientNetB3'  # Options: EfficientNetB3, MobileNetV2, ResNet50V2
INPUT_SHAPE = (300, 300, 3)  # EfficientNetB3: 300x300, MobileNetV2/ResNet50V2: 224x224
WEIGHTS = 'imagenet'

# Training configuration - Phase 1: Feature Extraction
PHASE1_EPOCHS = 10
PHASE1_BATCH_SIZE = 32
PHASE1_LEARNING_RATE = 1e-3
PHASE1_FREEZE_BASE = True

# Training configuration - Phase 2: Fine-Tuning
PHASE2_EPOCHS = 40
PHASE2_BATCH_SIZE = 32
PHASE2_LEARNING_RATE = 1e-5
PHASE2_UNFREEZE_LAYERS = 0.3  # Unfreeze last 30% of base layers

# Data augmentation
VALIDATION_SPLIT = 0.15
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
HORIZONTAL_FLIP = True
ZOOM_RANGE = 0.15
BRIGHTNESS_RANGE = [0.8, 1.2]

# Regularization
DROPOUT_RATE_1 = 0.4
DROPOUT_RATE_2 = 0.3
DENSE_UNITS = 256

# Callbacks
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
GRADCAM_DIR = os.path.join(OUTPUT_DIR, 'gradcam')

# Model saving
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.h5')
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, 'final_model.h5')
HISTORY_PATH = os.path.join(OUTPUT_DIR, 'training_history.pkl')

# Performance targets
TARGET_VAL_ACCURACY = 0.85
TARGET_TEST_ACCURACY = 0.82
TARGET_CARCINOMA_RECALL = 0.90

# Medical recommendations
RECOMMENDATIONS = {
    'Acne': {
        'description': 'Inflammatory skin condition affecting hair follicles and oil glands.',
        'care_advice': [
            'Wash affected areas twice daily with gentle cleanser',
            'Avoid picking or squeezing pimples',
            'Use non-comedogenic skincare products',
            'Consider over-the-counter benzoyl peroxide or salicylic acid',
            'Consult dermatologist if severe or persistent'
        ],
        'urgency': 'low'
    },
    'Carcinoma': {
        'description': 'Skin cancer requiring immediate medical attention.',
        'care_advice': [
            '⚠️ URGENT: Consult oncologist or dermatologist immediately',
            'Schedule biopsy for definitive diagnosis',
            'Avoid sun exposure and use high SPF sunscreen',
            'Do not attempt self-treatment',
            'Early detection significantly improves outcomes'
        ],
        'urgency': 'critical'
    },
    'Eczema': {
        'description': 'Chronic inflammatory condition causing itchy, red, dry skin.',
        'care_advice': [
            'Moisturize frequently with fragrance-free products',
            'Identify and avoid triggers (allergens, irritants)',
            'Use mild, soap-free cleansers',
            'Apply topical corticosteroids as prescribed',
            'Consider antihistamines for itching',
            'Consult dermatologist for persistent symptoms'
        ],
        'urgency': 'medium'
    },
    'Keratosis': {
        'description': 'Common benign skin growths, but may require monitoring.',
        'care_advice': [
            'Monitor for changes in size, color, or texture',
            'Protect from sun exposure (use SPF 30+)',
            'Consult dermatologist for removal options if desired',
            'Watch for signs of transformation (irregular borders, multiple colors)',
            'Regular skin examinations recommended'
        ],
        'urgency': 'low'
    },
    'Milia': {
        'description': 'Small white bumps caused by trapped keratin beneath skin surface.',
        'care_advice': [
            'Generally harmless and may resolve on their own',
            'Do not squeeze or pick at milia',
            'Gentle exfoliation may help prevent new formations',
            'Consult dermatologist for professional extraction if desired',
            'Use non-comedogenic skincare products'
        ],
        'urgency': 'low'
    },
    'Rosacea': {
        'description': 'Chronic inflammatory condition causing facial redness and visible blood vessels.',
        'care_advice': [
            'Identify and avoid personal triggers (spicy foods, alcohol, heat)',
            'Use gentle, fragrance-free skincare products',
            'Apply broad-spectrum sunscreen daily (SPF 30+)',
            'Consider topical medications (metronidazole, azelaic acid)',
            'Avoid harsh scrubs and astringents',
            'Consult dermatologist for persistent or severe cases'
        ],
        'urgency': 'medium'
    }
}

# API configuration
API_TITLE = "Skin Disease Classifier API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "AI-powered skin disease classification with explainability"
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for predictions
