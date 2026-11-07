
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# =====================================================
# CONFIGURATION
# =====================================================

SEED = 42
IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 8

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "LEAF_DATASET"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "medicinal_leaf_cnn_rnn.keras"
CLASS_NAMES_PATH = ARTIFACT_DIR / "class_names.json"

SELECTION_DIR = ARTIFACT_DIR / "checkpoints"
SELECTION_DIR.mkdir(parents=True, exist_ok=True)

PHASE1_PATH = SELECTION_DIR / "phase1_best.keras"
PHASE2_PATH = SELECTION_DIR / "phase2_best.keras"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =====================================================
# DATA LOADING
# =====================================================

def load_datasets(batch_size):
    for folder in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

    common = {
        "image_size": IMAGE_SIZE,
        "batch_size": batch_size,
        "label_mode": "int",
        "color_mode": "rgb",
    }

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        shuffle=True,
        seed=SEED,
        **common,
    )

    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        shuffle=False,
        **common,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        shuffle=False,
        **common,
    )

    if val_ds.class_names != class_names:
        raise ValueError("Validation classes differ from training classes.")

    if test_ds.class_names != class_names:
        raise ValueError("Test classes differ from training classes.")

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)

    autotune = tf.data.AUTOTUNE

    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
        class_names,
    )


# =====================================================
# CNN + RNN MODEL
# =====================================================

def build_model(num_classes):
    inputs = tf.keras.layers.Input(
        shape=(224, 224, 3),
        name="leaf_image",
    )

    # Data augmentation
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.06)(x)
    x = tf.keras.layers.RandomZoom(0.08)(x)
    x = tf.keras.layers.RandomTranslation(0.06, 0.06)(x)
    x = tf.keras.layers.RandomContrast(0.08)(x)

    # MobileNetV2 preprocessing: pixels scaled to [-1, 1]
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenet_preprocessing",
    )(x)

    # CNN feature extractor
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        name="mobilenetv2_backbone",
    )

    backbone.trainable = False
    features = backbone(x, training=False)

    # Convert feature maps into a sequence
    sequence = tf.keras.layers.Permute(
        (2, 1, 3),
        name="feature_map_permute",
    )(features)

    sequence = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten(),
        name="feature_rows",
    )(sequence)

    sequence = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(128, activation="relu"),
        name="feature_projection",
    )(sequence)

    sequence = tf.keras.layers.LayerNormalization(
        name="sequence_normalization",
    )(sequence)

    sequence = tf.keras.layers.Dropout(
        0.20,
        name="sequence_dropout",
    )(sequence)

    # RNN feature learning
    x = tf.keras.layers.SimpleRNN(
        units=64,
        activation="tanh",
        return_sequences=False,
        dropout=0.20,
        recurrent_dropout=0.0,
        name="leaf_simple_rnn",
    )(sequence)

    # Classification head
    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        name="classification_dense",
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="classification_batch_norm",
    )(x)

    x = tf.keras.layers.Dropout(
        0.35,
        name="classification_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)

    return tf.keras.Model(
        inputs,
        outputs,
        name="Leaf_MobileNetV2_SimpleRNN",
    )


# =====================================================
# CLASS WEIGHTS
# =====================================================

def calculate_class_weights(train_ds, num_classes):
    labels = np.concatenate(
        [y.numpy() for _, y in train_ds],
        axis=0,
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels,
    )

    return {
        int(i): float(weight)
        for i, weight in enumerate(weights)
    }


# =====================================================
# CALLBACKS
# =====================================================

def make_callbacks(checkpoint_path, csv_name, patience):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(ARTIFACT_DIR / csv_name),
            append=False,
        ),
    ]


# =====================================================
# TRAINING GRAPHS
# =====================================================

def save_plots(histories):
    train_acc, val_acc = [], []
    train_loss, val_loss = [], []

    for history in histories:
        train_acc.extend(history.history.get("accuracy", []))
        val_acc.extend(history.history.get("val_accuracy", []))
        train_loss.extend(history.history.get("loss", []))
        val_loss.extend(history.history.get("val_loss", []))

    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_acc, label="Training accuracy")
    plt.plot(epochs, val_acc, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN-RNN Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "accuracy_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN-RNN Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "loss_plot.png", dpi=150)
    plt.close()


# =====================================================
# TEST EVALUATION
# =====================================================

def evaluate_model(model, test_ds, class_names):
    y_true = np.concatenate(
        [y.numpy() for _, y in test_ds],
        axis=0,
    )

    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Test images: {len(y_true)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro precision: {precision * 100:.2f}%")
    print(f"Macro recall: {recall * 100:.2f}%")
    print(f"Macro F1-score: {f1 * 100:.2f}%")
    print("\nClassification report:\n")
    print(report)

    with open(
        ARTIFACT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("CNN-RNN LEAF CLASSIFICATION RESULTS\n\n")
        f.write(f"Test images: {len(y_true)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro precision: {precision:.4f}\n")
        f.write(f"Macro recall: {recall:.4f}\n")
        f.write(f"Macro F1-score: {f1:.4f}\n\n")
        f.write(report)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    size = max(10, len(class_names) * 0.45)
    fig, ax = plt.subplots(figsize=(size, size))

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    ).plot(
        ax=ax,
        xticks_rotation=90,
        colorbar=False,
        values_format="d",
    )

    plt.title("CNN-RNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        ARTIFACT_DIR / "confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    with open(
        ARTIFACT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "test_images": int(len(y_true)),
                "accuracy": float(accuracy),
                "macro_precision": float(precision),
                "macro_recall": float(recall),
                "macro_f1": float(f1),
            },
            f,
            indent=4,
        )


# =====================================================
# MAIN
# =====================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a MobileNetV2 + SimpleRNN leaf classifier."
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive.")

    if args.fine_tune_layers < 0:
        parser.error("Fine-tune layers cannot be negative.")

    print("TensorFlow version:", tf.__version__)
    print("Dataset:", DATA_DIR)
    print("Artifacts:", ARTIFACT_DIR)

    train_ds, val_ds, test_ds, class_names = load_datasets(
        args.batch_size
    )

    print("Number of classes:", len(class_names))

    class_weights = calculate_class_weights(
        train_ds,
        len(class_names),
    )

    model = build_model(len(class_names))

    # Phase 1: train the classification layers
    print("\nPHASE 1: FROZEN CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=make_callbacks(
            PHASE1_PATH,
            "phase1_history.csv",
            args.patience,
        ),
    )

    # Load the best Phase 1 model before fine-tuning.
    model = tf.keras.models.load_model(PHASE1_PATH)

    # Phase 2: fine-tune the CNN
    print("\nPHASE 2: FINE-TUNING CNN")

    backbone = model.get_layer("mobilenetv2_backbone")
    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    if args.fine_tune_layers > 0:
        for layer in backbone.layers[-args.fine_tune_layers:]:
            if not isinstance(
                layer,
                tf.keras.layers.BatchNormalization,
            ):
                layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=make_callbacks(
            PHASE2_PATH,
            "phase2_history.csv",
            args.patience,
        ),
    )

    save_plots([history1, history2])

    # Select checkpoint using validation loss, not test accuracy.
    best_path = None
    best_loss = float("inf")

    for path in (PHASE1_PATH, PHASE2_PATH):
        candidate = tf.keras.models.load_model(path)
        metrics = candidate.evaluate(
            val_ds,
            verbose=0,
            return_dict=True,
        )

        print(
            f"{path.name}: validation loss "
            f"{metrics['loss']:.4f}"
        )

        if float(metrics["loss"]) < best_loss:
            best_loss = float(metrics["loss"])
            best_path = path

        del candidate

    if best_path is None:
        raise RuntimeError("No valid checkpoint was created.")

    best_model = tf.keras.models.load_model(best_path)
    best_model.save(MODEL_PATH)

    print("\nSelected checkpoint:", best_path.name)
    print("Final model:", MODEL_PATH)

    # Evaluate test data once after model selection.
    evaluate_model(best_model, test_ds, class_names)

    print("\nTraining completed successfully.")
    print("Results saved in:", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
