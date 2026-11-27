import numpy as np
from keras.models import load_model as keras_load_model

from egg_hunt.config import CLASS_NAMES, RAW_DATA, MODEL_PATH
from egg_hunt.preprocessing import preprocess_batch, preprocess_image


def load_model(path):
    """
    Load a trained Keras model from a .keras file.
    """
    return keras_load_model(path)


def predict_single_image(model, data, top_k) -> dict:
    """
    Classify ONE cropped egg image (path, bytes, file object, PIL image or array).
    Returns the preprocessed image the model saw and the top_k species, best first.
    """
    img = preprocess_image(data)                       # (224, 224, 3) uint8
    probs = model.predict(img[np.newaxis, ...], verbose=0)[0]   # (11,)
    best = np.argsort(probs)[::-1][:top_k]
    return {
        "image": img,
        "predictions": [{"species": CLASS_NAMES[i], "confidence": float(probs[i])}
                        for i in best]
    }


def predict(model, data, top_k: int = 3) -> list[dict]:
    """
    Classify several cropped egg images.
    Returns one dict per image: the preprocessed image and its top_k species.
    """
    if not isinstance(data, (list, tuple)):
        data = [data]
    return [predict_single_image(model, d, top_k=top_k) for d in data]


if __name__ == "__main__":
    from egg_hunt.preprocessing import crop_egg, load_split

    # Dummy input: one egg from the test set, cropped the same way as in training
    test_df = load_split(RAW_DATA / "test/test_labels.json",
                         RAW_DATA / "test/test_data")
    row = test_df.sample(1).iloc[0]
    egg = crop_egg(row)

    model = load_model(MODEL_PATH)
    print(MODEL_PATH)

    result = predict(model, egg)[0]
    print(f"True label: {row['category_name']}")
    print(f"Image shape: {result['image'].shape}")
    for p in result["predictions"]:
        print(f"  {p['species']:<28} {p['confidence']:.1%}")
