from pathlib import Path

IMG_SIZE = 224
MARGIN = 0.3               # 20% margin around the egg on each side (training crops)
JITTER = 0.5               # how far the centre may move, as a share of the free margin space
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
MODELS_DIR   = PROJECT_ROOT / "models"
MODEL_PATH   = MODELS_DIR / "egg_classifier.keras"
CACHE = RAW_DATA / f"crops_{IMG_SIZE}_m{round(MARGIN)}_j{round(JITTER)}.npz"
