"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for model explainability
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap

    Args:
        img_array: Input image array (preprocessed)
        model: Trained model
        last_conv_layer_name: Name of last convolutional layer
        pred_index: Predicted class index (if None, use top prediction)

    Returns:
        heatmap: Grad-CAM heatmap
    """
    # Create a model that maps input to activations of last conv layer and output predictions
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradient of top predicted class with respect to activations of last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of output with respect to conv layer
    grads = tape.gradient(class_channel, conv_outputs)

    # Mean intensity of gradient over each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel by importance weight
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def find_last_conv_layer(model):
    """
    Automatically find the last convolutional layer in the model

    Args:
        model: Keras model

    Returns:
        str: Name of last convolutional layer
    """
    # Search for last Conv2D layer
    for layer in reversed(model.layers):
        # Check if layer has 4D output (batch, height, width, channels)
        if len(layer.output.shape) == 4:
            return layer.name

    raise ValueError("Could not find a 4D layer (Conv layer). Cannot apply Grad-CAM.")


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image

    Args:
        img_path: Path to original image
        heatmap: Grad-CAM heatmap
        cam_path: Path to save overlaid image
        alpha: Transparency of heatmap overlay

    Returns:
        superimposed_img: Image with Grad-CAM overlay
    """
    # Load original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create RGB image with jet colormap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    return superimposed_img


def generate_gradcam_visualization(model, img_path, save_path=None, class_name=None):
    """
    Generate complete Grad-CAM visualization with original, heatmap, and overlay

    Args:
        model: Trained model
        img_path: Path to image
        save_path: Path to save visualization
        class_name: Name of predicted class

    Returns:
        tuple: (heatmap, superimposed_img)
    """
    from src.data_pipeline import prepare_single_image

    # Prepare image
    img_array = prepare_single_image(img_path)

    # Get prediction
    predictions = model.predict(img_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = predictions[0][pred_class]

    if class_name is None:
        class_name = config.CLASSES[pred_class]

    # Find last conv layer
    last_conv_layer_name = find_last_conv_layer(model)
    print(f"Using layer '{last_conv_layer_name}' for Grad-CAM")

    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=pred_class)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    original_img = keras.preprocessing.image.load_img(img_path)
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Overlay
    superimposed_img = save_and_display_gradcam(
        img_path,
        heatmap,
        cam_path=save_path if save_path else "temp_gradcam.jpg",
        alpha=0.4
    )
    axes[2].imshow(superimposed_img)
    axes[2].set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Add prediction info
    fig.suptitle(
        f'Prediction: {class_name} (Confidence: {confidence:.2%})',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to {save_path}")

    plt.show()
    plt.close()

    return heatmap, superimposed_img


def batch_gradcam_visualization(model, image_paths, output_dir):
    """
    Generate Grad-CAM visualizations for multiple images

    Args:
        model: Trained model
        image_paths: List of image paths
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating Grad-CAM visualizations for {len(image_paths)} images...")
    print("="*70)

    for idx, img_path in enumerate(image_paths, 1):
        print(f"\nProcessing image {idx}/{len(image_paths)}: {os.path.basename(img_path)}")

        # Generate filename
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, f"{base_name}_gradcam.png")

        try:
            generate_gradcam_visualization(model, img_path, save_path=save_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print("\n" + "="*70)
    print(f"Batch Grad-CAM visualization complete!")
    print(f"Results saved to: {output_dir}")
    print("="*70 + "\n")


def compare_predictions_with_gradcam(model, image_paths, output_path=None):
    """
    Create a comparison grid of predictions with Grad-CAM overlays

    Args:
        model: Trained model
        image_paths: List of image paths
        output_path: Path to save comparison grid
    """
    from src.data_pipeline import prepare_single_image

    num_images = len(image_paths)
    cols = 3
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten() if num_images > 1 else [axes]

    for idx, img_path in enumerate(image_paths):
        # Prepare image and predict
        img_array = prepare_single_image(img_path)
        predictions = model.predict(img_array, verbose=0)
        pred_class = np.argmax(predictions[0])
        confidence = predictions[0][pred_class]
        class_name = config.CLASSES[pred_class]

        # Generate Grad-CAM
        last_conv_layer_name = find_last_conv_layer(model)
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=pred_class)

        # Create overlay
        temp_path = f"temp_overlay_{idx}.jpg"
        superimposed_img = save_and_display_gradcam(img_path, heatmap, cam_path=temp_path)

        # Plot
        axes[idx].imshow(superimposed_img)
        axes[idx].set_title(
            f"{class_name}\n{confidence:.2%}",
            fontsize=12,
            fontweight='bold'
        )
        axes[idx].axis('off')

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Hide empty subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(
        'Predictions with Grad-CAM Explainability',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison grid saved to {output_path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    """
    Test Grad-CAM module
    """
    import argparse
    from src.model import load_model

    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument('--model', type=str, default=config.BEST_MODEL_PATH, help='Path to model')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        sys.exit(1)

    # Load model
    print("Loading model...")
    model = load_model(args.model)

    # Generate Grad-CAM
    output_path = args.output if args.output else os.path.join(
        config.GRADCAM_DIR,
        f"{os.path.splitext(os.path.basename(args.image))[0]}_gradcam.png"
    )

    print("\nGenerating Grad-CAM visualization...")
    generate_gradcam_visualization(model, args.image, save_path=output_path)

    print("\nGrad-CAM generation complete!")
