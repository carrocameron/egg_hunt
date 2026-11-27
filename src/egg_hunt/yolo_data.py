from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from egg_hunt.config import CLASS_NAMES, RAW_DATA
from egg_hunt.preprocessing import load_image, load_split

YOLO_DIR = RAW_DATA / "yolo_data"      # raw_data/ is already git-ignored
LONG_SIDE = 1024


def export_split(df, split_name, long_side=LONG_SIDE):
    """Write the images and YOLO label files of one split."""
    images_dir = YOLO_DIR / "images" / split_name
    labels_dir = YOLO_DIR / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    mismatches = 0
    for image_id, group in tqdm(df.groupby("image_id"), desc=split_name):
        first = group.iloc[0]

        # Step 1: load the photo upright (same function as the crops pipeline)
        img = load_image(first["path"])
        img_w, img_h = img.size
        if (img_w, img_h) != (first["width"], first["height"]):
            mismatches += 1

        # Step 2: one label line per egg, clipped to the image, normalized to 0-1
        lines = []
        for _, egg in group.iterrows():
            x, y, w, h = egg["bbox"]
            x_min, y_min = max(0, x), max(0, y)
            x_max, y_max = min(img_w, x + w), min(img_h, y + h)
            if x_max <= x_min or y_max <= y_min:
                continue                     # box completely outside the image
            x_center = (x_min + x_max) / 2 / img_w
            y_center = (y_min + y_max) / 2 / img_h
            box_w = (x_max - x_min) / img_w
            box_h = (y_max - y_min) / img_h
            lines.append(f"{egg['category_id']} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        # Step 3: shrink the photo (aspect ratio kept) and save image and label
        scale = long_side / max(img_w, img_h)
        if scale < 1:
            new_size = (round(img_w * scale), round(img_h * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        stem = Path(first["file_name"]).stem
        img.save(images_dir / f"{stem}.jpg", quality=90)
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines))

    print(split_name, "| images:", df["image_id"].nunique(), "| size mismatches:", mismatches)


def write_data_yaml(colab_path="/content/yolo_data"):
    """The dataset description file YOLO reads. The path is where the data will be in Colab."""
    lines = [f"path: {colab_path}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    (YOLO_DIR / "data.yaml").write_text("\n".join(lines))


def build_yolo_dataset(val_size=0.2, seed=42):
    """Load the train labels, split by image, export everything."""
    train_df = load_split(RAW_DATA / "train/train_labels.json", RAW_DATA / "train/train_data")

    # Split by image, stratified, so the two-egg photos can't land on both sides
    image_table = train_df.groupby("image_id")["category_id"].first().reset_index()
    train_ids, val_ids = train_test_split(
        image_table["image_id"],
        test_size=val_size,
        stratify=image_table["category_id"],
        random_state=seed,
    )

    export_split(train_df[train_df["image_id"].isin(train_ids)], "train")
    export_split(train_df[train_df["image_id"].isin(val_ids)], "val")
    write_data_yaml()


if __name__ == "__main__":
    build_yolo_dataset()
