"""
Example usage script for Skin Disease Classifier

This script demonstrates how to use the classifier programmatically
without using command-line scripts.
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.model import load_model
from src.data_pipeline import prepare_single_image


def example_prediction():
    """
    Example: Load model and make a prediction
    """
    print("\n" + "="*70)
    print("EXAMPLE: SINGLE IMAGE PREDICTION")
    print("="*70)

    # Check if model exists
    model_path = config.BEST_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Please train the model first using: python src/train.py")
        return

    # Load model
    print("\n1. Loading model...")
    model = load_model(model_path)
    print("   ✓ Model loaded successfully")

    # Prepare image path (replace with your image)
    image_path = "path/to/your/image.jpg"

    if not os.path.exists(image_path):
        print(f"\n2. Example image not found at {image_path}")
        print("   Please provide a valid image path")
        return

    # Preprocess image
    print("\n2. Preprocessing image...")
    img_array = prepare_single_image(image_path)
    print(f"   ✓ Image preprocessed to shape: {img_array.shape}")

    # Make prediction
    print("\n3. Making prediction...")
    predictions = model.predict(img_array, verbose=0)

    # Get results
    pred_class_idx = np.argmax(predictions[0])
    pred_class_name = config.CLASSES[pred_class_idx]
    confidence = predictions[0][pred_class_idx]

    print(f"   ✓ Prediction complete")

    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nPredicted Disease: {pred_class_name}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

    print("\nAll Probabilities:")
    for i, class_name in enumerate(config.CLASSES):
        prob = predictions[0][i]
        bar = "█" * int(prob * 50)
        print(f"  {class_name:12s}: {prob:.4f} ({prob*100:5.2f}%) {bar}")

    # Get recommendation
    recommendation = config.RECOMMENDATIONS[pred_class_name]

    print("\n" + "="*70)
    print("MEDICAL RECOMMENDATION")
    print("="*70)
    print(f"\nDescription: {recommendation['description']}")
    print(f"Urgency: {recommendation['urgency'].upper()}")
    print("\nCare Advice:")
    for idx, advice in enumerate(recommendation['care_advice'], 1):
        print(f"  {idx}. {advice}")

    print("\n" + "="*70)
    print("DISCLAIMER")
    print("="*70)
    print("This is an AI prediction and should NOT replace professional medical advice.")
    print("="*70 + "\n")


def example_training():
    """
    Example: Training workflow
    """
    print("\n" + "="*70)
    print("EXAMPLE: TRAINING WORKFLOW")
    print("="*70)

    print("\nTo train the model, follow these steps:\n")

    print("1. Organize your dataset:")
    print("   dataset/")
    print("   ├── Acne/")
    print("   ├── Carcinoma/")
    print("   ├── Eczema/")
    print("   ├── Keratosis/")
    print("   ├── Milia/")
    print("   └── Rosacea/")

    print("\n2. Run training:")
    print("   python src/train.py --data-dir dataset/")

    print("\n3. Monitor with TensorBoard:")
    print("   tensorboard --logdir logs/")

    print("\n4. Evaluate model:")
    print("   python src/evaluate.py --model-path models/best_model.h5")

    print("\n5. Make predictions:")
    print("   python src/predict.py image.jpg --visualize --gradcam")

    print("\n6. Deploy API:")
    print("   python app.py")
    print("   Access at: http://localhost:8000")

    print("\n" + "="*70 + "\n")


def example_api_usage():
    """
    Example: API usage
    """
    print("\n" + "="*70)
    print("EXAMPLE: API USAGE")
    print("="*70)

    print("\nStart the API server:")
    print("  python app.py")

    print("\nThen use these endpoints:\n")

    print("1. Health Check:")
    print("   curl http://localhost:8000/health")

    print("\n2. Get Classes:")
    print("   curl http://localhost:8000/classes")

    print("\n3. Predict Image:")
    print('   curl -X POST "http://localhost:8000/predict" \\')
    print('        -F "file=@your_image.jpg"')

    print("\n4. Get Recommendation:")
    print("   curl http://localhost:8000/recommendation/Acne")

    print("\n5. Access Swagger UI:")
    print("   http://localhost:8000/docs")

    print("\n" + "="*70 + "\n")


def main():
    """
    Main function
    """
    print("\n" + "="*70)
    print("SKIN DISEASE CLASSIFIER - EXAMPLE USAGE")
    print("="*70)

    print("\nThis script demonstrates various usage examples.\n")

    print("Available examples:")
    print("  1. Single image prediction")
    print("  2. Training workflow")
    print("  3. API usage")

    try:
        choice = input("\nSelect example (1-3) or press Enter to see all: ").strip()

        if choice == "1":
            example_prediction()
        elif choice == "2":
            example_training()
        elif choice == "3":
            example_api_usage()
        else:
            example_prediction()
            example_training()
            example_api_usage()

    except KeyboardInterrupt:
        print("\n\nExiting...\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
