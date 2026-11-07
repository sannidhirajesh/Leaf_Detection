
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "medicinal_leaf_cnn_rnn.keras"
CLASS_NAMES_PATH = ROOT / "artifacts" / "class_names.json"
UPLOAD_DIR = ROOT / "static" / "uploads"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
IMAGE_SIZE = (224, 224)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

model = None
class_names = []


def load_resources():
    global model, class_names

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run train.py first."
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names not found: {CLASS_NAMES_PATH}. "
            "Run train.py first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "The number of model outputs does not match class_names.json."
        )


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def classify_image(image_path):
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb",
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]

    index = int(np.argmax(probabilities))

    return {
        "predicted_class": class_names[index],
        "confidence": round(float(probabilities[index]) * 100, 2),
        "top_predictions": [
            {
                "name": class_names[int(i)],
                "confidence": round(float(probabilities[i]) * 100, 2),
            }
            for i in np.argsort(probabilities)[::-1][:3]
        ],
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("leaf_image")

        if uploaded_file is None or not uploaded_file.filename:
            error = "Please select a leaf image."
        elif not allowed_file(uploaded_file.filename):
            error = "Upload a JPG, JPEG, PNG, or WEBP image."
        else:
            filename = secure_filename(uploaded_file.filename)
            destination = UPLOAD_DIR / filename

            # Avoid overwriting an existing upload.
            stem = destination.stem
            suffix = destination.suffix
            counter = 1

            while destination.exists():
                destination = UPLOAD_DIR / f"{stem}_{counter}{suffix}"
                counter += 1

            try:
                uploaded_file.save(destination)

                result = classify_image(str(destination))
                image_url = (
                    f"/static/uploads/{destination.name}"
                )
            except Exception as exc:
                app.logger.exception("Leaf prediction failed")
                error = f"Could not process this image: {exc}"

    return render_template(
        "index.html",
        result=result,
        image_url=image_url,
        error=error,
    )


if __name__ == "__main__":
    load_resources()
    app.run(debug=False, host="127.0.0.1", port=5000)
