"""Assemble a combined, balanced emotion dataset of target counts per class from multiple local sources.

Usage example:
  python scripts/create_combined_emotion_dataset.py \
    --sources data/face/FER2013 data/face/CK+ data/face/RAF-DB \
    --out data/face/emotion_dataset_5000 --counts-file scripts/emotion_counts.json --augment-if-needed --seed 42

Requires: you must download FER2013/CK+/RAF-DB manually (Kaggle links in README) and extract them under the provided source paths.

Behavior:
- For each target emotion, collects candidate images by searching under each source for subfolders named like the emotion (case-insensitive) or reading a mapping file if `/labels.csv` exists.
- If not enough images found, optionally augments existing images using `scripts/augment_images.py` to reach the target count.
- Avoids duplicates by checking file SHA1 hashes.
"""
import os
import argparse
import shutil
import random
import hashlib
import json
from glob import glob
from pathlib import Path

DEFAULT_COUNTS = {
    "angry": 720,
    "disgust": 600,
    "fear": 700,
    "happy": 900,
    "sad": 720,
    "surprise": 720,
    "neutral": 640,
}

IMG_EXTS = ('.png','.jpg','.jpeg')


def sha1_of_file(p):
    h = hashlib.sha1()
    with open(p, 'rb') as fh:
        while True:
            b = fh.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_images_for_emotion(source_dirs, emotion):
    res = []
    el = emotion.lower()
    for s in source_dirs:
        if not os.path.exists(s):
            continue
        # first: check for direct subdir named like emotion
        candidate = os.path.join(s, el)
        if os.path.isdir(candidate):
            for f in os.listdir(candidate):
                if f.lower().endswith(IMG_EXTS):
                    res.append(os.path.join(candidate, f))
        # fallback: search recursively for files where parent folder name contains emotion
        for f in glob(os.path.join(s, '**', '*.*'), recursive=True):
            if not f.lower().endswith(IMG_EXTS):
                continue
            parts = Path(f).parts
            if any(el in p.lower() for p in parts):
                res.append(f)
    # dedupe
    res = list(dict.fromkeys(res))
    return res


def ensure_target_counts(target_dir, counts, source_dirs, augment_if_needed=False, seed=42):
    random.seed(seed)
    os.makedirs(target_dir, exist_ok=True)
    hashes = set()

    for emotion, needed in counts.items():
        dst = os.path.join(target_dir, emotion)
        os.makedirs(dst, exist_ok=True)
        found = find_images_for_emotion(source_dirs, emotion)
        if not found:
            print('Warning: no candidates found for', emotion)
        # compute unique candidates by hash
        unique = []
        for f in found:
            try:
                h = sha1_of_file(f)
            except Exception:
                continue
            if h in hashes:
                continue
            unique.append((f,h))
        random.shuffle(unique)
        copied = 0
        for i,(f,h) in enumerate(unique):
            if copied >= needed:
                break
            name = f'{emotion}_{copied:05d}{Path(f).suffix.lower()}'
            dstp = os.path.join(dst, name)
            shutil.copyfile(f, dstp)
            hashes.add(h)
            copied += 1
        if copied < needed:
            print(f'Need {needed} for {emotion} but only copied {copied}.')
            if augment_if_needed and copied>0:
                # augment copies to reach needed using simple augmentations
                to_make = needed - copied
                print('Augmenting', to_make, 'images for', emotion)
                srcs = [os.path.join(dst, x) for x in os.listdir(dst) if x.lower().endswith(IMG_EXTS)]
                if not srcs:
                    print('No images to augment for', emotion)
                    continue
                # call augmentation in-process (simple flips/rotations/brightness)
                from PIL import Image, ImageEnhance
                funcs = [lambda im: im.transpose(Image.FLIP_LEFT_RIGHT), lambda im: im.rotate(10), lambda im: im.rotate(-10), lambda im: ImageEnhance.Brightness(im).enhance(0.9), lambda im: ImageEnhance.Brightness(im).enhance(1.1)]
                i = 0
                made = 0
                while made < to_make:
                    src = srcs[i % len(srcs)]
                    img = Image.open(src).convert('RGB')
                    f_aug = random.choice(funcs)
                    out = f_aug(img)
                    out_name = f'{emotion}_aug_{made:05d}.png'
                    out.save(os.path.join(dst, out_name))
                    made += 1; i += 1
                copied += made
                print('After augmentation copied:', copied)
            else:
                print('Not augmenting; consider using --augment-if-needed')
        print(f'Final count for {emotion}:', len([x for x in os.listdir(dst) if x.lower().endswith(IMG_EXTS)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', nargs='+', required=True, help='Source dataset directories (FER2013, CK+, RAF-DB)')
    parser.add_argument('--out', default='data/face/emotion_dataset_5000', help='Output target dir')
    parser.add_argument('--counts-file', default=None, help='Optional JSON file with per-emotion counts')
    parser.add_argument('--augment-if-needed', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.counts_file and os.path.exists(args.counts_file):
        counts = json.load(open(args.counts_file))
    else:
        counts = DEFAULT_COUNTS
    print('Using counts:', counts)

    ensure_target_counts(args.out, counts, args.sources, augment_if_needed=args.augment_if_needed, seed=args.seed)
    print('Dataset creation complete. Inspect', args.out)

if __name__ == '__main__':
    main()
