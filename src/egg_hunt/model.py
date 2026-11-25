from keras import Input
from keras.models import Sequential
from keras.layers import (
    RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness,
    Rescaling, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Flatten, GlobalAveragePooling2D, Dense, Dropout,
)
from keras.applications import EfficientNetB0
from keras.optimizers import Adam

from egg_hunt.config import IMG_SIZE, NUM_CLASSES


METRICS = ["accuracy", "Recall", "Precision", "AUC"]


# ------------------------------------------------------------------------------
# MODEL 1: Baseline CNN (same as test_baseline_cnn, so scores stay comparable)
# ------------------------------------------------------------------------------

def build_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES,
                augment=False, seed=None, lr=1e-3):
    """Raw uint8 images (n, 224, 224, 3) in -> class probabilities (n, 11) out."""
    model = Sequential(name="egg_classifier_cnn")
    model.add(Input(shape=(img_size, img_size, 3)))

    # Augmentation (training only, inactive in predict/evaluate, runs on 0-255)
    if augment:
        model.add(RandomFlip("horizontal_and_vertical", seed=seed))
        model.add(RandomRotation(0.25, fill_mode="constant", seed=seed))
        model.add(RandomZoom(0.1, fill_mode="constant", seed=seed))
        model.add(RandomContrast(0.2, seed=seed))
        model.add(RandomBrightness(0.2, value_range=(0, 255), seed=seed))

    model.add(Rescaling(1.0 / 255))        # crops are uint8 (0-255)

    # Block 1
    model.add(Conv2D(32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))

    # Block 2
    model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))

    # Block 3
    model.add(Conv2D(128, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))

    # Classifier
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",   # because y is one-hot
        metrics=METRICS,
    )
    return model


# ------------------------------------------------------------------------------
# MODEL 2: Deeper CNN with BatchNorm + GlobalAveragePooling
# ------------------------------------------------------------------------------

def build_model_global_average(img_size=IMG_SIZE, num_classes=NUM_CLASSES,
                               augment=False, seed=None, lr=1e-3):
    """Raw uint8 images in -> class probabilities out.
    4 conv blocks with BatchNorm, GlobalAveragePooling instead of Flatten."""
    model = Sequential(name="egg_classifier_cnn_gap")
    model.add(Input(shape=(img_size, img_size, 3)))

    # Augmentation (training only, inactive in predict/evaluate, runs on 0-255)
    if augment:
        model.add(RandomFlip("horizontal_and_vertical", seed=seed))
        model.add(RandomRotation(0.25, fill_mode="constant", seed=seed))
        model.add(RandomZoom(0.1, fill_mode="constant", seed=seed))
        model.add(RandomContrast(0.2, seed=seed))
        model.add(RandomBrightness(0.2, value_range=(0, 255), seed=seed))

    model.add(Rescaling(1.0 / 255))

    # Block 1
    model.add(Conv2D(32, kernel_size=3, padding="same", use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=2))

    # Block 2
    model.add(Conv2D(64, kernel_size=3, padding="same", use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=2))

    # Block 3
    model.add(Conv2D(128, kernel_size=3, padding="same", use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=2))

    # Block 4
    model.add(Conv2D(256, kernel_size=3, padding="same", use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=2))

    # Classifier
    model.add(GlobalAveragePooling2D())    # instead of Flatten -> 12.8M params gone
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=METRICS,
    )
    return model


# ------------------------------------------------------------------------------
# MODEL 3: EfficientNetB0 transfer learning
# ------------------------------------------------------------------------------

def build_model_efficientNet(img_size=IMG_SIZE, num_classes=NUM_CLASSES,
                             augment=False, seed=None, lr=1e-3):
    """Raw uint8 images (n, 224, 224, 3) in -> class probabilities out.
    EfficientNet does its own rescaling, so NO Rescaling(1/255) here."""
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False                 # phase 1: only train the new head

    model = Sequential(name="egg_classifier_efficientnet")
    model.add(Input(shape=(img_size, img_size, 3)))

    # Augmentation (training only, inactive in predict/evaluate, runs on 0-255)
    if augment:
        model.add(RandomFlip("horizontal_and_vertical", seed=seed))
        model.add(RandomRotation(0.25, fill_mode="constant", seed=seed))
        model.add(RandomZoom(0.1, fill_mode="constant", seed=seed))
        model.add(RandomContrast(0.2, seed=seed))
        model.add(RandomBrightness(0.2, value_range=(0, 255), seed=seed))

    # Pretrained feature extractor
    model.add(base)

    # Classifier
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=METRICS,
    )
    return model
