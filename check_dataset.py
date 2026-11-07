"""Validate the existing LEAF_DATASET train/validation/test folder structure."""
from pathlib import Path
from model_utils import SPLIT_DATA_DIR, IMAGE_EXTENSIONS


def main():
    print(f"Dataset path: {SPLIT_DATA_DIR}")
    if not SPLIT_DATA_DIR.is_dir():
        raise SystemExit("Dataset folder not found. Put LEAF_DATASET beside train.py.")
    all_classes = None
    total = 0
    for split in ("train", "validation", "test"):
        folder = SPLIT_DATA_DIR / split
        if not folder.is_dir():
            raise SystemExit(f"Missing split folder: {folder}")
        counts = {}
        for cls in sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith('.')):
            n = sum(1 for p in cls.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            if n:
                counts[cls.name] = n
        if len(counts) < 2:
            raise SystemExit(f"{split}: found fewer than 2 class folders containing images.")
        if all_classes is None:
            all_classes = set(counts)
        elif set(counts) != all_classes:
            raise SystemExit(f"Class folders differ in {split}. Expected {sorted(all_classes)}; found {sorted(counts)}")
        split_total = sum(counts.values())
        total += split_total
        print(f"\n{split}: {split_total} images across {len(counts)} classes")
        for name, count in counts.items():
            print(f"  {name}: {count}")
    print(f"\nDataset check passed: {len(all_classes)} classes, {total} images.")

if __name__ == '__main__':
    main()
