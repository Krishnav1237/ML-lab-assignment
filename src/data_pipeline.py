"""
Data preprocessing and augmentation pipeline
"""
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import config


def create_data_generators():
    """
    Create training, validation, and test data generators with augmentation

    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=config.ROTATION_RANGE,
        width_shift_range=config.WIDTH_SHIFT_RANGE,
        height_shift_range=config.HEIGHT_SHIFT_RANGE,
        horizontal_flip=config.HORIZONTAL_FLIP,
        zoom_range=config.ZOOM_RANGE,
        brightness_range=config.BRIGHTNESS_RANGE,
        fill_mode='nearest',
        validation_split=config.VALIDATION_SPLIT
    )

    # Test data generator (normalization only)
    test_datagen = ImageDataGenerator(rescale=1./255)

    print("\nData Pipeline Configuration:")
    print("="*70)
    print(f"Input Shape: {config.INPUT_SHAPE}")
    print(f"Batch Size: {config.PHASE1_BATCH_SIZE}")
    print(f"Validation Split: {config.VALIDATION_SPLIT}")
    print(f"Classes: {config.CLASSES}")
    print("\nAugmentation Settings:")
    print(f"  Rotation Range: {config.ROTATION_RANGE}°")
    print(f"  Width Shift Range: {config.WIDTH_SHIFT_RANGE}")
    print(f"  Height Shift Range: {config.HEIGHT_SHIFT_RANGE}")
    print(f"  Horizontal Flip: {config.HORIZONTAL_FLIP}")
    print(f"  Zoom Range: {config.ZOOM_RANGE}")
    print(f"  Brightness Range: {config.BRIGHTNESS_RANGE}")
    print("="*70 + "\n")

    return train_datagen, test_datagen


def load_data_from_directory(data_dir, train_datagen, test_datagen, batch_size=32):
    """
    Load data from directory structure

    Expected directory structure:
    data_dir/
        Acne/
            img1.jpg
            img2.jpg
        Carcinoma/
            img1.jpg
            img2.jpg
        ...

    Args:
        data_dir: Path to dataset directory
        train_datagen: Training data generator
        test_datagen: Test data generator
        batch_size: Batch size for training

    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            f"Please ensure your dataset is organized as:\n"
            f"  {data_dir}/\n"
            f"    Acne/\n"
            f"    Carcinoma/\n"
            f"    Eczema/\n"
            f"    Keratosis/\n"
            f"    Milia/\n"
            f"    Rosacea/\n"
        )

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=config.INPUT_SHAPE[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    # Validation generator
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=config.INPUT_SHAPE[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    print("\nDataset Loading Summary:")
    print("="*70)
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Class indices: {train_generator.class_indices}")
    print("="*70 + "\n")

    return train_generator, val_generator


def load_test_data_from_directory(test_dir, test_datagen, batch_size=32):
    """
    Load test data from separate directory

    Args:
        test_dir: Path to test dataset directory
        test_datagen: Test data generator
        batch_size: Batch size for testing

    Returns:
        test_generator: Test data generator
    """
    if not os.path.exists(test_dir):
        print(f"Warning: Test directory not found at {test_dir}")
        return None

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=config.INPUT_SHAPE[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    print(f"\nTest Dataset: {test_generator.samples} samples")

    return test_generator


def prepare_single_image(image_path):
    """
    Preprocess a single image for prediction

    Args:
        image_path: Path to image file

    Returns:
        numpy.ndarray: Preprocessed image ready for prediction
    """
    from tensorflow.keras.preprocessing import image as keras_image

    # Load image
    img = keras_image.load_img(
        image_path,
        target_size=config.INPUT_SHAPE[:2]
    )

    # Convert to array
    img_array = keras_image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def prepare_image_from_bytes(image_bytes):
    """
    Preprocess image from bytes (for API)

    Args:
        image_bytes: Image bytes

    Returns:
        numpy.ndarray: Preprocessed image ready for prediction
    """
    from PIL import Image
    import io
    from tensorflow.keras.preprocessing import image as keras_image

    # Load image from bytes
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize
    img = img.resize(config.INPUT_SHAPE[:2])

    # Convert to array
    img_array = keras_image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def get_dataset_statistics(data_dir):
    """
    Calculate and display dataset statistics

    Args:
        data_dir: Path to dataset directory
    """
    if not os.path.exists(data_dir):
        print(f"Dataset directory not found: {data_dir}")
        return

    print("\nDataset Statistics:")
    print("="*70)

    total_images = 0
    class_counts = {}

    for class_name in config.CLASSES:
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = count
            total_images += count
            print(f"  {class_name}: {count} images")

    print(f"\nTotal Images: {total_images}")
    print(f"Classes: {len(class_counts)}")
    print(f"Average per class: {total_images / len(class_counts):.1f}")

    # Check balance
    if len(set(class_counts.values())) == 1:
        print("Dataset is perfectly balanced ✓")
    else:
        imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
        print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    print("="*70 + "\n")

    return class_counts


def visualize_augmentation(data_generator, num_samples=9):
    """
    Visualize data augmentation examples

    Args:
        data_generator: Data generator
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt

    # Get a batch of images
    images, labels = next(data_generator)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()

    for idx in range(min(num_samples, len(images))):
        axes[idx].imshow(images[idx])
        class_idx = np.argmax(labels[idx])
        axes[idx].set_title(f"{config.CLASSES[class_idx]}", fontweight='bold')
        axes[idx].axis('off')

    plt.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    plt.close()

    print("Data augmentation visualization complete")


if __name__ == "__main__":
    """
    Test data pipeline
    """
    print("Testing Data Pipeline...")

    # Get statistics
    if os.path.exists(config.DATA_DIR):
        get_dataset_statistics(config.DATA_DIR)

        # Create generators
        train_datagen, test_datagen = create_data_generators()

        # Load data
        train_gen, val_gen = load_data_from_directory(
            config.DATA_DIR,
            train_datagen,
            test_datagen,
            batch_size=config.PHASE1_BATCH_SIZE
        )

        # Visualize augmentation
        print("\nGenerating augmentation examples...")
        visualize_augmentation(train_gen)

        print("\nData pipeline test complete!")
    else:
        print(f"\nDataset not found at {config.DATA_DIR}")
        print("Please organize your dataset first.")
