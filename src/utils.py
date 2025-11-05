"""
Utility functions for visualization, plotting, and helpers
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import config


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss

    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()
    plt.close()


def plot_classification_report(y_true, y_pred, classes, save_path=None):
    """
    Generate and plot classification report

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        save_path: Path to save the report
    """
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    # Print text report
    print("\nClassification Report:")
    print("=" * 70)
    print(classification_report(y_true, y_pred, target_names=classes))

    # Create dataframe for visualization
    metrics_data = []
    for class_name in classes:
        if class_name in report:
            metrics_data.append([
                class_name,
                report[class_name]['precision'],
                report[class_name]['recall'],
                report[class_name]['f1-score']
            ])

    # Plot metrics
    metrics = np.array(metrics_data)
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    precision_bars = ax.bar(x - width, [float(m[1]) for m in metrics], width, label='Precision', alpha=0.8)
    recall_bars = ax.bar(x, [float(m[2]) for m in metrics], width, label='Recall', alpha=0.8)
    f1_bars = ax.bar(x + width, [float(m[3]) for m in metrics], width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)

    add_value_labels(precision_bars)
    add_value_labels(recall_bars)
    add_value_labels(f1_bars)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Classification report plot saved to {save_path}")

    plt.show()
    plt.close()

    return report


def check_carcinoma_recall(y_true, y_pred, classes):
    """
    Check if Carcinoma recall meets the target threshold

    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names

    Returns:
        bool: Whether recall meets target
    """
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    carcinoma_recall = report['Carcinoma']['recall']

    print(f"\n{'='*70}")
    print(f"CRITICAL METRIC CHECK - Carcinoma Recall")
    print(f"{'='*70}")
    print(f"Current Recall: {carcinoma_recall:.4f}")
    print(f"Target Recall: {config.TARGET_CARCINOMA_RECALL:.4f}")

    if carcinoma_recall >= config.TARGET_CARCINOMA_RECALL:
        print(f"✓ TARGET MET - Excellent performance on cancer detection!")
    else:
        print(f"✗ TARGET NOT MET - Consider model improvements")
    print(f"{'='*70}\n")

    return carcinoma_recall >= config.TARGET_CARCINOMA_RECALL


def save_history(history, path):
    """
    Save training history to file

    Args:
        history: Training history object
        path: Path to save the history
    """
    with open(path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to {path}")


def load_history(path):
    """
    Load training history from file

    Args:
        path: Path to load the history from

    Returns:
        dict: Training history
    """
    with open(path, 'rb') as f:
        history = pickle.load(f)
    print(f"Training history loaded from {path}")
    return history


def ensure_directories():
    """
    Ensure all required directories exist
    """
    directories = [
        config.MODEL_DIR,
        config.LOG_DIR,
        config.OUTPUT_DIR,
        config.GRADCAM_DIR
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("All required directories verified/created")


def print_model_summary(model):
    """
    Print detailed model summary

    Args:
        model: Keras model
    """
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    model.summary()
    print("="*70 + "\n")

    # Count trainable parameters
    trainable_count = sum([np.prod(w.shape) for w in model.trainable_weights])
    non_trainable_count = sum([np.prod(w.shape) for w in model.non_trainable_weights])

    print(f"Total parameters: {trainable_count + non_trainable_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")
    print("="*70 + "\n")


def get_class_weights(y_train):
    """
    Calculate class weights for handling imbalance

    Args:
        y_train: Training labels

    Returns:
        dict: Class weights
    """
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weight_dict = dict(enumerate(class_weights))

    print("\nClass Weights:")
    for idx, (class_name, weight) in enumerate(zip(config.CLASSES, class_weights)):
        print(f"  {class_name}: {weight:.4f}")

    return class_weight_dict


def plot_sample_predictions(model, generator, num_samples=9):
    """
    Plot sample predictions with true labels

    Args:
        model: Trained model
        generator: Data generator
        num_samples: Number of samples to plot
    """
    images, labels = next(generator)
    predictions = model.predict(images[:num_samples])

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()

    for idx in range(min(num_samples, len(images))):
        axes[idx].imshow(images[idx])

        true_label = config.CLASSES[np.argmax(labels[idx])]
        pred_label = config.CLASSES[np.argmax(predictions[idx])]
        confidence = np.max(predictions[idx])

        color = 'green' if true_label == pred_label else 'red'

        axes[idx].set_title(
            f"True: {true_label}\nPred: {pred_label} ({confidence:.2%})",
            color=color,
            fontweight='bold'
        )
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()


def print_evaluation_summary(history, test_accuracy, report):
    """
    Print comprehensive evaluation summary

    Args:
        history: Training history
        test_accuracy: Test accuracy
        report: Classification report
    """
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)

    # Training metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])

    print(f"\nTraining Metrics:")
    print(f"  Final Train Accuracy: {final_train_acc:.4f}")
    print(f"  Final Val Accuracy: {final_val_acc:.4f}")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")

    # Target comparison
    print(f"\nTarget Comparison:")
    print(f"  Val Accuracy Target: {config.TARGET_VAL_ACCURACY:.2f} | Achieved: {best_val_acc:.4f} {'✓' if best_val_acc >= config.TARGET_VAL_ACCURACY else '✗'}")
    print(f"  Test Accuracy Target: {config.TARGET_TEST_ACCURACY:.2f} | Achieved: {test_accuracy:.4f} {'✓' if test_accuracy >= config.TARGET_TEST_ACCURACY else '✗'}")

    # Critical metric
    carcinoma_recall = report['Carcinoma']['recall']
    print(f"  Carcinoma Recall Target: {config.TARGET_CARCINOMA_RECALL:.2f} | Achieved: {carcinoma_recall:.4f} {'✓' if carcinoma_recall >= config.TARGET_CARCINOMA_RECALL else '✗'}")

    print("="*70 + "\n")
