from keras import Input, Sequential, layers

from egg_hunt.config import IMG_SIZE, NUM_CLASSES


def augmentation_block(seed=None) -> Sequential:
    """Random changes during training only. Eggs have no 'up' under a microscope,
    so flips and rotations are safe. Contrast covers staining/lighting differences.
    Runs on 0–255 values, i.e. BEFORE rescaling."""
    return Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomRotation(0.25, fill_mode="constant", seed=seed),  # black fill, like our padding
        layers.RandomZoom(0.1, fill_mode="constant", seed=seed),
        layers.RandomContrast(0.2, seed=seed),
    ], name="augmentation")


def preprocessing_block() -> Sequential:
    return Sequential([layers.Rescaling(1.0 / 255)], name="preprocessing")


def cnn_block() -> Sequential:
    """Same conv blocks as test_baseline_cnn, so scores stay comparable."""
    return Sequential([
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
    ], name="cnn")


def classifier_block(num_classes: int = NUM_CLASSES) -> Sequential:
    return Sequential([
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="classifier")

def build_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES, augment=False, seed=None):
    """Raw uint8 images (n, 224, 224, 3) in -> class probabilities (n, 11) out."""

    model = Sequential(name="egg_classifier")

    model.add(Input(shape=(img_size, img_size, 3)))

    if augment:
        model.add(augmentation_block(seed))

    model.add(preprocessing_block())
    model.add(cnn_block())
    model.add(classifier_block(num_classes))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", "Recall", "Precision", "AUC"]
        )

    return model
