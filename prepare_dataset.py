"""Organize an extracted Mendeley dataset into data/leaves/<species>/*.jpg.

Extract the downloaded archive into data/raw first. This script searches nested
folders for species directories containing images and copies images into a clean
class-folder layout. It does not rename the species labels beyond making folder
names safe for Windows.
"""
from pathlib import Path
import re, shutil, sys
from PIL import Image, UnidentifiedImageError
from model_utils import ROOT, RAW_DIR, DATA_DIR, IMAGE_EXTENSIONS


def safe_name(s):
    s = re.sub(r'[<>:"/\\|?*]', '_', s.strip())
    s = re.sub(r'\s+', '_', s).strip(' ._')
    return s or 'unknown_species'


def count_direct_images(folder):
    try:
        return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    except OSError:
        return []


def find_class_folders():
    # A class directory should contain images itself; avoid treating an archive's
    # top-level container as a class when it only contains other directories.
    candidates = []
    for folder in RAW_DIR.rglob('*'):
        if folder.is_dir():
            imgs = count_direct_images(folder)
            if len(imgs) >= 2:
                candidates.append((folder, imgs))
    if not candidates:
        return []
    # Prefer the shallowest level that has at least two sibling image folders.
    by_parent = {}
    for folder, imgs in candidates:
        by_parent.setdefault(folder.parent, []).append((folder, imgs))
    good_parents = [(parent, items) for parent, items in by_parent.items() if len(items) >= 2]
    if good_parents:
        # Use the parent group with the most class-like folders, breaking ties by shallowest path.
        parent, items = sorted(good_parents, key=lambda pair: (-len(pair[1]), len(pair[0].parts)))[0]
        return items
    # Fallback to all candidates at the shallowest image-folder depth.
    min_depth = min(len(folder.parts) for folder, _ in candidates)
    return [(f, imgs) for f, imgs in candidates if len(f.parts) == min_depth]


def main():
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print('No extracted files found. Download the dataset from https://data.mendeley.com/datasets/nnytj2v3n5/1')
        print(f'Extract the ZIP into: {RAW_DIR}')
        sys.exit(1)
    found = find_class_folders()
    if len(found) < 2:
        print('Could not automatically identify at least two class folders.')
        print('Expected species folders containing image files under data/raw/.')
        print('You can manually copy each species folder into data/leaves/<species_name>/.')
        sys.exit(1)
    if DATA_DIR.exists():
        for p in DATA_DIR.iterdir():
            if p.is_dir(): shutil.rmtree(p)
            else: p.unlink()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total, skipped = 0, 0
    print('Detected candidate species folders:')
    for folder, images in sorted(found, key=lambda x: x[0].name.lower()):
        label = safe_name(folder.name)
        out = DATA_DIR / label
        out.mkdir(parents=True, exist_ok=True)
        written = 0
        for i, source in enumerate(images):
            try:
                with Image.open(source) as im:
                    im.convert('RGB').save(out / f'{label}_{i:05d}.jpg', quality=94)
                written += 1
            except (OSError, UnidentifiedImageError):
                skipped += 1
        if written == 0:
            shutil.rmtree(out, ignore_errors=True)
        else:
            total += written
            print(f'  {label}: {written} images')
    classes = [p for p in DATA_DIR.iterdir() if p.is_dir() and any(p.iterdir())]
    if len(classes) < 2:
        print('Preparation finished, but fewer than two usable classes were found.')
        sys.exit(1)
    print(f'\nPrepared {total} images across {len(classes)} classes in {DATA_DIR}')
    if skipped: print(f'Skipped unreadable files: {skipped}')
    print('Next: python train.py')

if __name__ == '__main__': main()
