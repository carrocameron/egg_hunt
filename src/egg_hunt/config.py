from pathlib import Path

IMG_SIZE = 224
MARGIN = 0.3               # 20% margin around the egg on each side (training crops)
JITTER = 1  # how far the centre may move, as a share of the free margin space
SEED = 42
PAD_COLOR = (0, 0, 0)      # same black PIL uses when a bbox crop runs past the image edge

# Fixed list in category_id order – like taxifare's fixed min/max values,
# independent of the training data and not dependent on a gitignored JSON.
CLASS_NAMES = [
    "Ascaris lumbricoides",
    "Capillaria philippinensis",
    "Enterobius vermicularis",
    "Fasciolopsis buski",
    "Hookworm egg",
    "Hymenolepis diminuta",
    "Hymenolepis nana",
    "Opisthorchis viverrine",
    "Paragonimus spp",
    "Taenia spp. egg",
    "Trichuris trichiura",
]
NUM_CLASSES = len(CLASS_NAMES)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA     = PROJECT_ROOT / "raw_data"
CLASS_MODELS_DIR = PROJECT_ROOT / "class_models"   # Keras classifiers
YOLO_MODELS_DIR  = PROJECT_ROOT / "yolo_models"    # YOLO detectors
#MODEL_PATH   = MODELS_DIR / "egg_classifier.keras"
CACHE = RAW_DATA / f"crops_{IMG_SIZE}_m{round(MARGIN*100)}%_j{round(JITTER*100)}%.npz"

def _latest_model(models_dir: Path, pattern: str, default_name: str) -> Path:
    """Return the most recently modified file matching `pattern` in models_dir."""
    models = sorted(models_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    # Fall back to a default name so importing config never crashes on an empty folder
    return models[-1] if models else models_dir / default_name

CLASS_MODEL_PATH = _latest_model(CLASS_MODELS_DIR, "*.keras", "egg_classifier.keras")
YOLO_MODEL_PATH  = _latest_model(YOLO_MODELS_DIR,  "*.pt",    "egg_detector.pt")
