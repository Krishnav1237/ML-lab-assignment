"""
Training script with two-phase strategy
Phase 1: Feature Extraction (frozen base model)
Phase 2: Fine-Tuning (unfrozen last layers)
"""
import os
import sys
import argparse
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model import build_model, compile_model, get_callbacks, unfreeze_model
from src.data_pipeline import create_data_generators, load_data_from_directory, get_dataset_statistics
from src.utils import (
    ensure_directories,
    print_model_summary,
    plot_training_history,
    save_history,
    get_class_weights
)


def train_phase1(model, train_gen, val_gen, class_weights=None):
    """
    Phase 1: Feature Extraction
    Train only the top layers with frozen base model

    Args:
        model: Keras model
        train_gen: Training data generator
        val_gen: Validation data generator
        class_weights: Class weights for imbalanced data

    Returns:
        history: Training history
    """
    print("\n" + "="*70)
    print("PHASE 1: FEATURE EXTRACTION")
    print("="*70)
    print(f"Epochs: {config.PHASE1_EPOCHS}")
    print(f"Batch Size: {config.PHASE1_BATCH_SIZE}")
    print(f"Learning Rate: {config.PHASE1_LEARNING_RATE}")
    print(f"Base Model: FROZEN")
    print("="*70 + "\n")

    # Compile model
    model = compile_model(
        model,
        learning_rate=config.PHASE1_LEARNING_RATE,
        phase='phase1'
    )

    # Get callbacks
    callbacks = get_callbacks(phase='phase1')

    # Calculate steps
    steps_per_epoch = train_gen.samples // config.PHASE1_BATCH_SIZE
    validation_steps = val_gen.samples // config.PHASE1_BATCH_SIZE

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}\n")

    # Train
    history = model.fit(
        train_gen,
        epochs=config.PHASE1_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1
    )

    print("\n" + "="*70)
    print("PHASE 1 COMPLETE")
    print("="*70)
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print("="*70 + "\n")

    return history


def train_phase2(model, base_model, train_gen, val_gen, class_weights=None):
    """
    Phase 2: Fine-Tuning
    Unfreeze last layers and train with lower learning rate

    Args:
        model: Keras model
        base_model: Base model
        train_gen: Training data generator
        val_gen: Validation data generator
        class_weights: Class weights for imbalanced data

    Returns:
        history: Training history
    """
    print("\n" + "="*70)
    print("PHASE 2: FINE-TUNING")
    print("="*70)
    print(f"Epochs: {config.PHASE2_EPOCHS}")
    print(f"Batch Size: {config.PHASE2_BATCH_SIZE}")
    print(f"Learning Rate: {config.PHASE2_LEARNING_RATE}")
    print(f"Unfreeze Ratio: {config.PHASE2_UNFREEZE_LAYERS*100:.0f}%")
    print("="*70 + "\n")

    # Unfreeze layers
    model = unfreeze_model(
        model,
        base_model,
        unfreeze_ratio=config.PHASE2_UNFREEZE_LAYERS
    )

    # Recompile with lower learning rate
    model = compile_model(
        model,
        learning_rate=config.PHASE2_LEARNING_RATE,
        phase='phase2'
    )

    # Get callbacks
    callbacks = get_callbacks(phase='phase2')

    # Calculate steps
    steps_per_epoch = train_gen.samples // config.PHASE2_BATCH_SIZE
    validation_steps = val_gen.samples // config.PHASE2_BATCH_SIZE

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}\n")

    # Train
    history = model.fit(
        train_gen,
        epochs=config.PHASE2_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1
    )

    print("\n" + "="*70)
    print("PHASE 2 COMPLETE")
    print("="*70)
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print("="*70 + "\n")

    return history


def combine_histories(history1, history2):
    """
    Combine training histories from both phases

    Args:
        history1: Phase 1 history
        history2: Phase 2 history

    Returns:
        combined_history: Combined history object
    """
    class CombinedHistory:
        def __init__(self):
            self.history = {}

    combined = CombinedHistory()

    for key in history1.history.keys():
        combined.history[key] = history1.history[key] + history2.history[key]

    return combined


def main(args):
    """
    Main training function

    Args:
        args: Command line arguments
    """
    # Ensure directories exist
    ensure_directories()

    print("\n" + "="*70)
    print("SKIN DISEASE CLASSIFIER - TRAINING")
    print("="*70)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Classes: {config.CLASSES}")
    print(f"Input Shape: {config.INPUT_SHAPE}")
    print(f"Dataset: {args.data_dir}")
    print("="*70 + "\n")

    # Get dataset statistics
    if args.show_stats:
        get_dataset_statistics(args.data_dir)

    # Create data generators
    train_datagen, test_datagen = create_data_generators()

    # Load data
    train_gen, val_gen = load_data_from_directory(
        args.data_dir,
        train_datagen,
        test_datagen,
        batch_size=config.PHASE1_BATCH_SIZE
    )

    # Calculate class weights
    class_weights = None
    if args.use_class_weights:
        # Get all labels from training generator
        y_train = train_gen.classes
        class_weights = get_class_weights(y_train)

    # Build model
    model, base_model = build_model(
        model_name=args.model,
        freeze_base=True
    )

    # Print model summary
    if args.show_summary:
        print_model_summary(model)

    # Phase 1: Feature Extraction
    if not args.skip_phase1:
        history1 = train_phase1(model, train_gen, val_gen, class_weights)

        # Save Phase 1 model
        phase1_path = os.path.join(config.MODEL_DIR, 'phase1_model.h5')
        model.save(phase1_path)
        print(f"Phase 1 model saved to {phase1_path}")

        # Save Phase 1 history
        phase1_history_path = os.path.join(config.OUTPUT_DIR, 'phase1_history.pkl')
        save_history(history1, phase1_history_path)

        # Plot Phase 1 history
        if args.plot_history:
            phase1_plot_path = os.path.join(config.OUTPUT_DIR, 'phase1_training_history.png')
            plot_training_history(history1, save_path=phase1_plot_path)
    else:
        print("Skipping Phase 1...")
        history1 = None

    # Phase 2: Fine-Tuning
    if not args.skip_phase2:
        # Reload batch generators for phase 2
        train_gen, val_gen = load_data_from_directory(
            args.data_dir,
            train_datagen,
            test_datagen,
            batch_size=config.PHASE2_BATCH_SIZE
        )

        history2 = train_phase2(model, base_model, train_gen, val_gen, class_weights)

        # Save Phase 2 model
        phase2_path = os.path.join(config.MODEL_DIR, 'phase2_model.h5')
        model.save(phase2_path)
        print(f"Phase 2 model saved to {phase2_path}")

        # Save Phase 2 history
        phase2_history_path = os.path.join(config.OUTPUT_DIR, 'phase2_history.pkl')
        save_history(history2, phase2_history_path)

        # Plot Phase 2 history
        if args.plot_history:
            phase2_plot_path = os.path.join(config.OUTPUT_DIR, 'phase2_training_history.png')
            plot_training_history(history2, save_path=phase2_plot_path)
    else:
        print("Skipping Phase 2...")
        history2 = None

    # Combine and save histories
    if history1 and history2:
        combined_history = combine_histories(history1, history2)
        save_history(combined_history, config.HISTORY_PATH)

        if args.plot_history:
            combined_plot_path = os.path.join(config.OUTPUT_DIR, 'combined_training_history.png')
            plot_training_history(combined_history, save_path=combined_plot_path)

    # Save final model
    model.save(config.FINAL_MODEL_PATH)
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Final model saved to: {config.FINAL_MODEL_PATH}")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")
    print(f"Training history saved to: {config.HISTORY_PATH}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Skin Disease Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default=config.DATA_DIR,
        help='Path to dataset directory'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=config.MODEL_NAME,
        choices=['EfficientNetB3', 'MobileNetV2', 'ResNet50V2'],
        help='Base model architecture'
    )

    parser.add_argument(
        '--skip-phase1',
        action='store_true',
        help='Skip Phase 1 (Feature Extraction)'
    )

    parser.add_argument(
        '--skip-phase2',
        action='store_true',
        help='Skip Phase 2 (Fine-Tuning)'
    )

    parser.add_argument(
        '--use-class-weights',
        action='store_true',
        help='Use class weights for training'
    )

    parser.add_argument(
        '--show-stats',
        action='store_true',
        default=True,
        help='Show dataset statistics'
    )

    parser.add_argument(
        '--show-summary',
        action='store_true',
        default=True,
        help='Show model summary'
    )

    parser.add_argument(
        '--plot-history',
        action='store_true',
        default=True,
        help='Plot training history'
    )

    args = parser.parse_args()

    # Run training
    main(args)
