import io
import json
import os
from pathlib import Path
from typing import BinaryIO, Iterable, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from keras.utils import to_categorical

from egg_hunt.config import CACHE, IMG_SIZE, MARGIN, NUM_CLASSES, PAD_COLOR, JITTER, SEED

ImageSource = Union[str, Path, bytes, BinaryIO, Image.Image, np.ndarray]


# =====================================================================
# 1) SHARED IMAGE PIPELINE  – used in training AND for user uploads
# =====================================================================

def load_image(source: ImageSource) -> Image.Image:
    """Open whatever we get (path, raw bytes, file object, PIL image, array)
    and return an upright RGB PIL image."""
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(source.astype(np.uint8))
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    else:  # path, or file-like (FastAPI UploadFile.file, Streamlit UploadedFile)
        img = Image.open(source)
    img = ImageOps.exif_transpose(img)   # phone photos come rotated otherwise
    return img.convert("RGB")


def preprocess_image(source: ImageSource, img_size: int = IMG_SIZE) -> np.ndarray:
    """Cropped egg image -> (img_size, img_size, 3) uint8 array, ready for the model.

    Uses padding instead of stretching: training crops are square, but a user's
    crop usually isn't, and stretching would distort the egg's shape."""
    img = load_image(source)
    img = ImageOps.pad(img, (img_size, img_size), color=PAD_COLOR)
    return np.asarray(img, dtype=np.uint8)


def preprocess_batch(sources: Iterable[ImageSource], img_size: int = IMG_SIZE) -> np.ndarray:
    """Several images -> (n, img_size, img_size, 3)."""
    return np.stack([preprocess_image(s, img_size) for s in sources])


# =====================================================================
# 2) TRAINING DATA PIPELINE  – needs labels + bounding boxes
# =====================================================================

def load_split(labels_path, data_path) -> pd.DataFrame:
    with open(labels_path) as f:
        data = json.load(f)

    images      = pd.DataFrame(data["images"]).rename(columns={"id": "image_id"})
    annotations = pd.DataFrame(data["annotations"]).rename(columns={"id": "egg_id"})
    categories  = pd.DataFrame(data["categories"]).rename(
        columns={"id": "category_id", "name": "category_name"})

    df = (images
          .merge(annotations, on="image_id")
          .merge(categories, on="category_id")
          .drop(columns=["license", "coco_url", "supercategory"]))
    df["path"] = df["file_name"].apply(lambda n: os.path.join(data_path, n))
    return df


def crop_from_bbox(img: Image.Image, bbox, margin: float = MARGIN,
                   jitter: float = 0.0, rng: np.random.Generator | None = None) -> Image.Image:
    """Square crop around a COCO bbox [x, y, w, h] with a margin.
    This is the step a user does by hand before uploading. It also works with
    a bbox from a detector (e.g. YOLO) if you add detection later."""
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * margin)

    if jitter > 0:
        max_dx = (side - w) / 2 * jitter      # free space left/right of the egg
        max_dy = (side - h) / 2 * jitter      # free space above/below the egg
        cx += rng.uniform(-max_dx, max_dx)
        cy += rng.uniform(-max_dy, max_dy)

    side = int(round(side))
    left = int(round(cx - side / 2))
    top  = int(round(cy - side / 2))
    return img.crop((left, top, left + side, top + side))


def crop_egg(row: pd.Series, margin: float = MARGIN,
             jitter: float = 0.0, rng: np.random.Generator | None = None) -> Image.Image:
    """One dataframe row -> cropped egg (PIL, not resized). Good for viewing."""
    return crop_from_bbox(load_image(row["path"]), row["bbox"], margin, jitter, rng=rng)


def build_arrays(df: pd.DataFrame, img_size: int = IMG_SIZE,
                 jitter: float = 0.0, seed: int | None = SEED):

    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)                     # row i in df == X[i]
    X = np.zeros((len(df), img_size, img_size, 3), dtype=np.uint8)
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Building arrays"):
        X[i] = preprocess_image(crop_egg(row, jitter=jitter, ring=rng), img_size)   # <- same function the app uses
    y = to_categorical(df["category_id"], num_classes=NUM_CLASSES)
    return X, y


def get_cropped_arrays(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       cache_path=CACHE, img_size: int = IMG_SIZE,
                       jitter_train: bool = True,
                       jitter_test: bool = False,
                       jitter: float = JITTER):
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    if jitter_train:
        X_train, y_train = build_arrays(train_df, img_size, jitter=jitter, seed=SEED)
    else: X_train, y_train = build_arrays(train_df, img_size)

    if jitter_test:
        X_test,  y_test  = build_arrays(test_df,  img_size, jitter=jitter, seed=SEED)
    else: X_test,  y_test  = build_arrays(test_df,  img_size)

    np.savez(cache_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return X_train, y_train, X_test, y_test
