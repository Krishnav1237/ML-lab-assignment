"""
Prediction script for single image inference
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model import load_model
from src.data_pipeline import prepare_single_image
from src.gradcam import generate_gradcam_visualization


def predict_image(model, image_path, show_probabilities=True):
    """
    Predict disease class for a single image

    Args:
        model: Trained model
        image_path: Path to image
        show_probabilities: Whether to show all class probabilities

    Returns:
        dict: Prediction results
    """
    # Prepare image
    img_array = prepare_single_image(image_path)

    # Make prediction
    predictions = model.predict(img_array, verbose=0)

    # Get results
    pred_class_idx = np.argmax(predictions[0])
    pred_class_name = config.CLASSES[pred_class_idx]
    confidence = predictions[0][pred_class_idx]

    # Create results dictionary
    results = {
        'predicted_class': pred_class_name,
        'predicted_index': int(pred_class_idx),
        'confidence': float(confidence),
        'all_probabilities': {
            config.CLASSES[i]: float(predictions[0][i])
            for i in range(len(config.CLASSES))
        },
        'recommendation': config.RECOMMENDATIONS[pred_class_name]
    }

    # Print results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"\nPredicted Disease: {pred_class_name}")
    print(f"Confidence: {confidence:.2%}")

    # Check if confidence is above threshold
    if confidence < config.CONFIDENCE_THRESHOLD:
        print(f"\n⚠️  WARNING: Low confidence prediction (< {config.CONFIDENCE_THRESHOLD:.0%})")
        print("    Consider getting multiple opinions or professional consultation")

    if show_probabilities:
        print("\nAll Class Probabilities:")
        sorted_probs = sorted(
            results['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for class_name, prob in sorted_probs:
            bar = "█" * int(prob * 50)
            print(f"  {class_name:12s}: {prob:.4f} ({prob*100:5.2f}%) {bar}")

    # Print medical recommendation
    rec = results['recommendation']
    print("\n" + "="*70)
    print("MEDICAL INFORMATION")
    print("="*70)
    print(f"\nCondition: {pred_class_name}")
    print(f"Description: {rec['description']}")
    print(f"Urgency: {rec['urgency'].upper()}")

    print("\nCare Advice:")
    for idx, advice in enumerate(rec['care_advice'], 1):
        print(f"  {idx}. {advice}")

    print("\n" + "="*70)
    print("DISCLAIMER")
    print("="*70)
    print("This is an AI-powered prediction and should NOT replace professional")
    print("medical diagnosis. Please consult a qualified dermatologist or healthcare")
    print("provider for proper diagnosis and treatment.")
    print("="*70 + "\n")

    return results


def visualize_prediction(model, image_path, results):
    """
    Visualize prediction with probabilities bar chart

    Args:
        model: Trained model
        image_path: Path to image
        results: Prediction results
    """
    from tensorflow.keras.preprocessing import image as keras_image

    # Load original image
    img = keras_image.load_img(image_path)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot image
    axes[0].imshow(img)
    axes[0].set_title(
        f"Predicted: {results['predicted_class']}\nConfidence: {results['confidence']:.2%}",
        fontsize=14,
        fontweight='bold'
    )
    axes[0].axis('off')

    # Plot probabilities
    classes = list(results['all_probabilities'].keys())
    probs = list(results['all_probabilities'].values())

    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    classes_sorted = [classes[i] for i in sorted_indices]
    probs_sorted = [probs[i] for i in sorted_indices]

    colors = ['green' if c == results['predicted_class'] else 'steelblue' for c in classes_sorted]

    axes[1].barh(classes_sorted, probs_sorted, color=colors, alpha=0.8)
    axes[1].set_xlabel('Probability', fontsize=12, fontweight='bold')
    axes[1].set_title('Class Probabilities', fontsize=14, fontweight='bold')
    axes[1].set_xlim([0, 1])
    axes[1].grid(axis='x', alpha=0.3)

    # Add value labels
    for idx, (c, p) in enumerate(zip(classes_sorted, probs_sorted)):
        axes[1].text(p + 0.02, idx, f'{p:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.close()


def batch_predict(model, image_dir, output_csv=None):
    """
    Predict multiple images in a directory

    Args:
        model: Trained model
        image_dir: Directory containing images
        output_csv: Optional path to save results as CSV

    Returns:
        list: List of prediction results
    """
    import glob
    import pandas as pd

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    if not image_paths:
        print(f"No images found in {image_dir}")
        return []

    print(f"\nProcessing {len(image_paths)} images from {image_dir}...")
    print("="*70)

    results_list = []

    for idx, img_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")

        try:
            # Prepare and predict
            img_array = prepare_single_image(img_path)
            predictions = model.predict(img_array, verbose=0)

            pred_class_idx = np.argmax(predictions[0])
            pred_class_name = config.CLASSES[pred_class_idx]
            confidence = predictions[0][pred_class_idx]

            result = {
                'filename': os.path.basename(img_path),
                'predicted_class': pred_class_name,
                'confidence': confidence,
                **{f'prob_{cls}': predictions[0][i] for i, cls in enumerate(config.CLASSES)}
            }

            results_list.append(result)

            print(f"  Prediction: {pred_class_name} ({confidence:.2%})")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print("\n" + "="*70)
    print(f"Batch prediction complete! Processed {len(results_list)} images.")
    print("="*70 + "\n")

    # Save to CSV if requested
    if output_csv and results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}\n")

    return results_list


def main(args):
    """
    Main prediction function

    Args:
        args: Command line arguments
    """
    # Load model
    print(f"\nLoading model from {args.model}...")
    model = load_model(args.model)

    if args.batch:
        # Batch prediction
        output_csv = os.path.join(config.OUTPUT_DIR, 'batch_predictions.csv')
        batch_predict(model, args.image, output_csv=output_csv)

    else:
        # Single image prediction
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            sys.exit(1)

        # Make prediction
        results = predict_image(
            model,
            args.image,
            show_probabilities=args.show_probs
        )

        # Visualize
        if args.visualize:
            print("\nGenerating visualization...")
            visualize_prediction(model, args.image, results)

        # Generate Grad-CAM
        if args.gradcam:
            print("\nGenerating Grad-CAM explanation...")
            gradcam_path = os.path.join(
                config.GRADCAM_DIR,
                f"{os.path.splitext(os.path.basename(args.image))[0]}_gradcam.png"
            )
            generate_gradcam_visualization(model, args.image, save_path=gradcam_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict skin disease from image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'image',
        type=str,
        help='Path to image file or directory (for batch mode)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=config.BEST_MODEL_PATH,
        help='Path to trained model'
    )

    parser.add_argument(
        '--show-probs',
        action='store_true',
        default=True,
        help='Show all class probabilities'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        default=False,
        help='Show prediction visualization'
    )

    parser.add_argument(
        '--gradcam',
        action='store_true',
        default=False,
        help='Generate Grad-CAM explainability visualization'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        default=False,
        help='Batch prediction mode (process directory)'
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train a model first using train.py")
        sys.exit(1)

    # Run prediction
    main(args)
