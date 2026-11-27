"""
Image functions used for inference (API) AND training.
Only needs numpy + PIL, so the API doesn't load pandas or other training libraries.
"""
import io
from pathlib import Path
from typing import BinaryIO, Iterable, Union

import numpy as np
from PIL import Image, ImageOps

from egg_hunt.config import IMG_SIZE, MARGIN, PAD_COLOR

ImageSource = Union[str, Path, bytes, BinaryIO, Image.Image, np.ndarray]


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
