"""Re-run evaluation on the exact held-out test split saved during training."""
import json, sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from model_utils import ARTIFACT_DIR, MODEL_PATH, SPLIT_PATH, load_class_names, IMAGE_SIZE
from train import make_dataset


def main():
    if not MODEL_PATH.exists() or not SPLIT_PATH.exists():
        raise SystemExit('Model/test split not found. Run python train.py first.')
    with SPLIT_PATH.open(encoding='utf-8') as f: split = json.load(f)
    names = load_class_names()
    ds = make_dataset(np.array(split['test_paths']), np.array(split['test_labels']), 12)
    model = tf.keras.models.load_model(MODEL_PATH)
    y = np.array(split['test_labels'])
    p = model.predict(ds, verbose=0).argmax(axis=1)
    pr, re, f1, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    print(f'Test accuracy: {accuracy_score(y,p):.2%}')
    print(f'Macro precision: {pr:.4f} | Macro recall: {re:.4f} | Macro F1: {f1:.4f}')
    print(classification_report(y, p, labels=list(range(len(names))), target_names=names, zero_division=0))

if __name__ == '__main__': main()
