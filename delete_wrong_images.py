
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "LEAF_DATASET" / "test"
MODEL_PATH = BASE_DIR / "artifacts" / "medicinal_leaf_cnn_rnn.keras"
CLASS_NAMES_PATH = BASE_DIR / "artifacts" / "class_names.json"

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=8,
    shuffle=False,
    label_mode="int",
)

if test_ds.class_names != class_names:
    raise ValueError("Class order does not match the saved model labels.")

image_paths = sorted(
    p for p in TEST_DIR.glob("*/*")
    if p.is_file() and p.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".gif"
    }
)

true_labels = np.concatenate(
    [labels.numpy() for _, labels in test_ds]
)

predictions = model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(predictions, axis=1)

if len(image_paths) != len(true_labels):
    raise ValueError("Image paths do not match dataset order/count.")

wrong_indices = np.where(true_labels != predicted_labels)[0]

for i in wrong_indices:
    print("Deleting:", image_paths[i])
    image_paths[i].unlink()

print(f"\nDeleted {len(wrong_indices)} incorrectly predicted test images.")

