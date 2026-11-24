from keras import Input, Sequential, layers, applications, optimizers

from egg_hunt.config import IMG_SIZE, NUM_CLASSES


def augmentation_block(seed=None) -> Sequential:
    """Random changes during training only (inactive in predict/evaluate).
    Runs on 0–255 values."""
    return Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomRotation(0.25, fill_mode="constant", seed=seed),
        layers.RandomZoom(0.1, fill_mode="constant", seed=seed),
        layers.RandomContrast(0.2, seed=seed),
        layers.RandomBrightness(0.2, value_range=(0, 255), seed=seed),  # NEW: different microscopes/stains
    ], name="augmentation")


def rescaling_block() -> Sequential:
    return Sequential([layers.Rescaling(1.0 / 255)], name="rescaling")


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

def conv_block(filters):
    return [
        layers.Conv2D(filters, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2),
    ]

def classifier_block(num_classes: int = NUM_CLASSES) -> Sequential:
    return Sequential([
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="classifier")

def build_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES,
                augment=False, seed=None, lr=1e-3):
    """Raw uint8 images (n, 224, 224, 3) in -> class probabilities (n, 11) out."""

    model = Sequential(name="egg_classifier_cnn")

    model.add(Input(shape=(img_size, img_size, 3)))

    if augment:
        model.add(augmentation_block(seed))

    model.add(rescaling_block())
    model.add(cnn_block())
    model.add(classifier_block(num_classes))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", "Recall", "Precision", "AUC"]
        )

    return model

def build_model_v2(img_size=IMG_SIZE, num_classes=NUM_CLASSES,
                augment=False, seed=None, lr=1e-3):

    model = Sequential(name="egg_classifier_cnn")

    model.add(Input(shape=(img_size, img_size, 3)))

    if augment:
        model.add(augmentation_block(seed))

    model.add(layers.Rescaling(1.0 / 255))

    for f in (32, 64, 128, 256):                 # 4 blocks instead of 3
        for layer in conv_block(f):
            model.add(layer)

    model.add(layers.GlobalAveragePooling2D())   # instead of Flatten -> 12.8M params gone
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", "Recall", "Precision", "AUC"]
        )

    return model

def build_transfer_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES,
                         augment=False, seed=None, lr=1e-3):
    """Raw uint8 images (n, 224, 224, 3) in -> class probabilities out.
    EfficientNet does its own rescaling, so NO Rescaling(1/255) here."""

    base = applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False            # phase 1: only train the new head

    model = Sequential(name="egg_classifier")

    model.add(Input(shape=(img_size, img_size, 3)))

    if augment:
        model.add(augmentation_block(seed))

    model.add(base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", "Recall", "Precision", "AUC"]
        )

    return model
