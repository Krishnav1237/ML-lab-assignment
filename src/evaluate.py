"""
Model evaluation script with comprehensive metrics
"""
import os
import sys
import argparse
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model import load_model
from src.data_pipeline import (
    create_data_generators,
    load_data_from_directory,
    load_test_data_from_directory
)
from src.utils import (
    plot_confusion_matrix,
    plot_classification_report,
    check_carcinoma_recall,
    load_history,
    print_evaluation_summary,
    plot_sample_predictions
)


def evaluate_model(model, data_generator, dataset_name='Test'):
    """
    Evaluate model on dataset

    Args:
        model: Trained model
        data_generator: Data generator
        dataset_name: Name of dataset for display

    Returns:
        tuple: (loss, accuracy, y_true, y_pred)
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING ON {dataset_name.upper()} SET")
    print(f"{'='*70}\n")

    # Reset generator
    data_generator.reset()

    # Get predictions
    print("Generating predictions...")
    predictions = model.predict(data_generator, verbose=1)

    # Get true labels
    y_true = data_generator.classes
    y_pred = np.argmax(predictions, axis=1)

    # Calculate metrics
    loss, accuracy = model.evaluate(data_generator, verbose=0)

    print(f"\n{dataset_name} Results:")
    print(f"{'='*70}")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*70}\n")

    return loss, accuracy, y_true, y_pred


def main(args):
    """
    Main evaluation function

    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("SKIN DISEASE CLASSIFIER - EVALUATION")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Classes: {config.CLASSES}")
    print("="*70 + "\n")

    # Load model
    model = load_model(args.model_path)

    # Create data generators
    train_datagen, test_datagen = create_data_generators()

    # Determine which dataset to evaluate
    if args.test_dir and os.path.exists(args.test_dir):
        # Separate test directory
        print(f"Using separate test directory: {args.test_dir}")
        test_gen = load_test_data_from_directory(
            args.test_dir,
            test_datagen,
            batch_size=32
        )
    else:
        # Use validation split from training directory
        print(f"Using validation split from: {args.data_dir}")
        _, test_gen = load_data_from_directory(
            args.data_dir,
            train_datagen,
            test_datagen,
            batch_size=32
        )

    # Evaluate model
    loss, accuracy, y_true, y_pred = evaluate_model(
        model,
        test_gen,
        dataset_name='Test'
    )

    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, config.CLASSES, save_path=cm_path)

    # Plot classification report
    print("\nGenerating classification report...")
    report_path = os.path.join(config.OUTPUT_DIR, 'classification_report.png')
    report = plot_classification_report(y_true, y_pred, config.CLASSES, save_path=report_path)

    # Check critical metric: Carcinoma recall
    check_carcinoma_recall(y_true, y_pred, config.CLASSES)

    # Load and display training history if available
    if args.show_history and os.path.exists(config.HISTORY_PATH):
        print("\nLoading training history...")
        history_dict = load_history(config.HISTORY_PATH)

        # Create a simple object to hold history
        class HistoryHolder:
            def __init__(self, history_dict):
                self.history = history_dict

        history = HistoryHolder(history_dict)

        # Print evaluation summary
        print_evaluation_summary(history, accuracy, report)
    else:
        print("\nTraining history not available")

    # Sample predictions visualization
    if args.show_samples:
        print("\nGenerating sample predictions...")
        plot_sample_predictions(model, test_gen, num_samples=9)

    # Calculate per-class accuracy
    print("\nPer-Class Accuracy:")
    print("="*70)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    for idx, (class_name, acc) in enumerate(zip(config.CLASSES, per_class_acc)):
        status = "✓" if acc >= 0.80 else "✗"
        print(f"  {class_name:12s}: {acc:.4f} ({acc*100:.2f}%) {status}")

    print("="*70)

    # Check if targets are met
    print("\n" + "="*70)
    print("TARGET ACHIEVEMENT")
    print("="*70)

    targets_met = []

    # Test accuracy target
    test_target_met = accuracy >= config.TARGET_TEST_ACCURACY
    targets_met.append(test_target_met)
    print(f"Test Accuracy:     {accuracy:.4f} >= {config.TARGET_TEST_ACCURACY:.4f} {'✓' if test_target_met else '✗'}")

    # Carcinoma recall target
    carcinoma_recall = report['Carcinoma']['recall']
    carcinoma_target_met = carcinoma_recall >= config.TARGET_CARCINOMA_RECALL
    targets_met.append(carcinoma_target_met)
    print(f"Carcinoma Recall:  {carcinoma_recall:.4f} >= {config.TARGET_CARCINOMA_RECALL:.4f} {'✓' if carcinoma_target_met else '✗'}")

    print("="*70)

    if all(targets_met):
        print("\n🎉 All targets achieved! Model is ready for deployment.")
    else:
        print("\n⚠️  Some targets not met. Consider further training or model improvements.")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {config.OUTPUT_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Skin Disease Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default=config.BEST_MODEL_PATH,
        help='Path to trained model'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default=config.DATA_DIR,
        help='Path to dataset directory (for validation split)'
    )

    parser.add_argument(
        '--test-dir',
        type=str,
        default=None,
        help='Path to separate test dataset directory'
    )

    parser.add_argument(
        '--show-history',
        action='store_true',
        default=True,
        help='Show training history'
    )

    parser.add_argument(
        '--show-samples',
        action='store_true',
        default=True,
        help='Show sample predictions'
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first using train.py")
        sys.exit(1)

    # Run evaluation
    main(args)
