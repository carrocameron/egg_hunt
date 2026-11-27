import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from keras.utils import to_categorical

from egg_hunt.config import CACHE, IMG_SIZE, MARGIN, NUM_CLASSES, JITTER, SEED
from egg_hunt.image_utils import (ImageSource, load_image, crop_from_bbox,
                                  preprocess_image, preprocess_batch)


# =====================================================================
# TRAINING DATA PIPELINE  – needs labels + bounding boxes
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
        X[i] = preprocess_image(crop_egg(row, jitter=jitter, rng=rng), img_size)   # <- same function the app uses
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
