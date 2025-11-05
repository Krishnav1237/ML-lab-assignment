"""
CNN Model Architecture with Transfer Learning
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3, MobileNetV2, ResNet50V2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)
import config


def build_model(model_name='EfficientNetB3', freeze_base=True):
    """
    Build CNN model with transfer learning

    Args:
        model_name: Name of base model (EfficientNetB3, MobileNetV2, ResNet50V2)
        freeze_base: Whether to freeze base model layers

    Returns:
        tuple: (model, base_model)
    """
    print(f"\nBuilding {model_name} model...")
    print("="*70)

    # Select base model
    if model_name == 'EfficientNetB3':
        base_model = EfficientNetB3(
            include_top=False,
            weights=config.WEIGHTS,
            input_shape=config.INPUT_SHAPE
        )
    elif model_name == 'MobileNetV2':
        # MobileNetV2 uses 224x224
        input_shape = (224, 224, 3)
        base_model = MobileNetV2(
            include_top=False,
            weights=config.WEIGHTS,
            input_shape=input_shape
        )
    elif model_name == 'ResNet50V2':
        # ResNet50V2 uses 224x224
        input_shape = (224, 224, 3)
        base_model = ResNet50V2(
            include_top=False,
            weights=config.WEIGHTS,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Freeze or unfreeze base model
    base_model.trainable = not freeze_base

    # Build custom head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(name='global_avg_pool'),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(config.DROPOUT_RATE_1, name='dropout_1'),
        layers.Dense(config.DENSE_UNITS, activation='relu', name='dense_1'),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(config.DROPOUT_RATE_2, name='dropout_2'),
        layers.Dense(config.NUM_CLASSES, activation='softmax', name='output')
    ], name=f'{model_name}_SkinDiseaseClassifier')

    print(f"Base Model: {model_name}")
    print(f"Input Shape: {config.INPUT_SHAPE}")
    print(f"Base Model Trainable: {not freeze_base}")
    print(f"Number of Classes: {config.NUM_CLASSES}")
    print(f"Dense Units: {config.DENSE_UNITS}")
    print(f"Dropout Rates: {config.DROPOUT_RATE_1}, {config.DROPOUT_RATE_2}")
    print("="*70 + "\n")

    return model, base_model


def unfreeze_model(model, base_model, unfreeze_ratio=0.3):
    """
    Unfreeze the last N% of base model layers for fine-tuning

    Args:
        model: Complete model
        base_model: Base model
        unfreeze_ratio: Ratio of layers to unfreeze (0.3 = last 30%)

    Returns:
        model: Model with unfrozen layers
    """
    # Calculate number of layers to unfreeze
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * (1 - unfreeze_ratio))

    print(f"\nUnfreezing model for fine-tuning...")
    print("="*70)
    print(f"Total base model layers: {total_layers}")
    print(f"Unfreezing from layer: {unfreeze_from} ({unfreeze_ratio*100:.0f}%)")

    # Freeze all layers first
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    # Unfreeze last layers
    for layer in base_model.layers[unfreeze_from:]:
        layer.trainable = True

    # Count trainable parameters
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])

    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")
    print("="*70 + "\n")

    return model


def compile_model(model, learning_rate, phase='phase1'):
    """
    Compile model with optimizer and loss

    Args:
        model: Keras model
        learning_rate: Learning rate
        phase: Training phase name

    Returns:
        model: Compiled model
    """
    print(f"\nCompiling model for {phase}...")
    print("="*70)
    print(f"Learning Rate: {learning_rate}")
    print(f"Optimizer: Adam")
    print(f"Loss: Categorical Crossentropy")
    print(f"Metrics: Accuracy")
    print("="*70 + "\n")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_callbacks(phase='phase1'):
    """
    Get training callbacks

    Args:
        phase: Training phase name

    Returns:
        list: List of callbacks
    """
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),

        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1,
            mode='min'
        ),

        # Model checkpoint
        ModelCheckpoint(
            config.BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),

        # TensorBoard logging
        TensorBoard(
            log_dir=f"{config.LOG_DIR}/{phase}",
            histogram_freq=1,
            write_graph=True
        )
    ]

    print(f"\nCallbacks configured for {phase}:")
    print("="*70)
    print(f"  - EarlyStopping (patience={config.EARLY_STOPPING_PATIENCE})")
    print(f"  - ReduceLROnPlateau (patience={config.REDUCE_LR_PATIENCE}, factor={config.REDUCE_LR_FACTOR})")
    print(f"  - ModelCheckpoint (save_best_only=True)")
    print(f"  - TensorBoard (log_dir={config.LOG_DIR}/{phase})")
    print("="*70 + "\n")

    return callbacks


def create_cosine_decay_schedule(initial_lr, total_steps):
    """
    Create cosine decay learning rate schedule

    Args:
        initial_lr: Initial learning rate
        total_steps: Total training steps

    Returns:
        schedule: Learning rate schedule
    """
    schedule = optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps
    )

    return schedule


def load_model(model_path):
    """
    Load trained model

    Args:
        model_path: Path to model file

    Returns:
        model: Loaded model
    """
    print(f"\nLoading model from {model_path}...")

    model = models.load_model(model_path)

    print("Model loaded successfully!")
    print("="*70)
    model.summary()
    print("="*70 + "\n")

    return model


if __name__ == "__main__":
    """
    Test model architecture
    """
    print("Testing Model Architecture...")

    # Build model
    model, base_model = build_model(
        model_name=config.MODEL_NAME,
        freeze_base=True
    )

    # Compile model
    model = compile_model(
        model,
        learning_rate=config.PHASE1_LEARNING_RATE,
        phase='test'
    )

    # Print summary
    print("\nModel Summary:")
    print("="*70)
    model.summary()
    print("="*70)

    # Count parameters
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])

    print(f"\nTotal parameters: {trainable_count + non_trainable_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")

    # Test unfreezing
    print("\n\nTesting layer unfreezing...")
    model = unfreeze_model(model, base_model, unfreeze_ratio=config.PHASE2_UNFREEZE_LAYERS)

    print("\nModel architecture test complete!")
