
import json
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "medicinal_leaf_cnn_rnn.keras"
CLASS_NAMES_PATH = ROOT / "artifacts" / "class_names.json"


def load_model_and_classes():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run train.py first."
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names not found: {CLASS_NAMES_PATH}\n"
            "Run train.py first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return model, class_names


def predict_leaf(image_path, model, class_names):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load and resize image to the model input dimensions.
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224),
        color_mode="rgb",
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # The model includes its own preprocessing layers.
    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    # Display the three highest-scoring classes.
    top_indices = np.argsort(probabilities)[::-1][:3]

    print("\n========== LEAF PREDICTION ==========")
    print(f"Image: {image_path.name}")
    print(f"Predicted class: {predicted_class}")
    print(f"Model confidence: {confidence:.2f}%")

    print("\nTop predictions:")
    for index in top_indices:
        print(
            f"{class_names[int(index)]}: "
            f"{float(probabilities[index]) * 100:.2f}%"
        )

    print("=====================================")

    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Predict a leaf class from an image."
    )
    parser.add_argument(
        "image",
        help="Path to the leaf image to classify.",
    )
    args = parser.parse_args()

    model, class_names = load_model_and_classes()
    predict_leaf(args.image, model, class_names)


if __name__ == "__main__":
    main()
