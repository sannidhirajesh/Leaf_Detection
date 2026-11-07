# Medicinal Leaf AI — CNN + SimpleRNN (ready for your split dataset)

This project classifies a leaf image into one of the classes present in your dataset. It uses MobileNetV2 as the CNN feature extractor, a SimpleRNN over spatial feature rows, and Streamlit for the demo interface.

## 1. Put your dataset here

Extract the dataset inside this project folder, beside `train.py`, with the exact folder name `LEAF_DATASET`:

```text
medicinal_leaf_cnn_simple_rnn/
├── LEAF_DATASET/
│   ├── train/<class-name>/*.jpg
│   ├── validation/<class-name>/*.jpg
│   └── test/<class-name>/*.jpg
├── train.py
├── evaluate.py
├── model_utils.py
├── app.py
└── requirements.txt
```

The code also accepts the earlier folder name `Medicinal_Leaf_Dataset`. Each split must have the same class subfolder names. Supported images: JPG, JPEG, PNG, BMP, WEBP, TIF, TIFF. The dataset is not included in this ZIP; keep your own dataset folder beside the code.

## 2. Recommended Windows setup

Use Python 3.12. In VS Code terminal, from this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. Select `.venv` via `Ctrl+Shift+P` → `Python: Select Interpreter`.

If you already installed packages into global Python, you can use that interpreter, but a virtual environment avoids conflicts with unrelated packages. The project requirements are bounded to TensorFlow 2.x and NumPy 1.x for compatibility.

## 3. Verify the dataset before training

```powershell
python check_dataset.py
```

This only checks folder/class/image counts; it does not train or alter images.

## 4. Train

```powershell
python train.py --batch-size 8 --epochs 10
```

`--epochs 10` means up to 10 epochs for each of two stages; early stopping can stop sooner. CPU training may take a while. The first run may download MobileNetV2 ImageNet weights, so internet is needed for that download unless the weights are already cached.

Training reads the existing `train/` and `validation/` splits and evaluates on `test/`. It does not re-split the dataset. Outputs go into `artifacts/`, including the trained `.keras` model, class names, metrics, learning curves, and confusion matrix. Accuracy is only known after the run completes.

## 5. Evaluate and launch the app

```powershell
python evaluate.py
streamlit run app.py
```

Run the app command after training. Use the local URL shown in the terminal.

## Troubleshooting

- Dataset not found: ensure `LEAF_DATASET` is at the same level as `train.py`, not inside `data/` and not nested twice (e.g. `LEAF_DATASET/LEAF_DATASET/train`).
- Missing class: every class folder under `train/` must also exist under `validation/` and `test/`.
- Memory error: stop training and retry with `--batch-size 4`.
- TensorFlow informational oneDNN messages are not errors.
- This is a learning prototype, not medical advice or guaranteed botanical identification; it can only predict classes represented in training.
