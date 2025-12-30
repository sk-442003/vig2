"""Convert FER-2013 CSV into per-emotion image folders.

Usage:
  python scripts/fer2013_to_images.py --csv path/to/fer2013.csv --out data/face/FER2013 --sample 3000

Writes images to `--out/<emotion>/` where emotion names follow the standard mapping:
0: angry, 1: disgust, 2: fear, 3: happy, 4: sad, 5: surprise, 6: neutral

If --sample is provided (int), it will randomly sample up to that many images total 
(or you can set --per_emotion N to cap per-class samples).
"""
import os
import argparse
import csv
import numpy as np
from PIL import Image
import random

FER_MAP = {
    '0': 'angry',
    '1': 'disgust',
    '2': 'fear',
    '3': 'happy',
    '4': 'sad',
    '5': 'surprise',
    '6': 'neutral',
}


def ensure_dirs(out_dir):
    for v in FER_MAP.values():
        d = os.path.join(out_dir, v)
        os.makedirs(d, exist_ok=True)


def pixels_to_image(pixels_str, size=(48,48)):
    pixels = np.fromstring(pixels_str, dtype=int, sep=' ')
    if pixels.size != size[0] * size[1]:
        raise ValueError('Unexpected pixel count')
    arr = pixels.reshape(size).astype('uint8')
    return Image.fromarray(arr, mode='L')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to fer2013.csv')
    parser.add_argument('--out', default='data/face/FER2013', help='Output directory to write per-emotion folders')
    parser.add_argument('--sample', type=int, default=None, help='Total number of images to sample (approx, proportional)')
    parser.add_argument('--per_emotion', type=int, default=None, help='Max images per emotion')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f'CSV not found: {args.csv}')

    ensure_dirs(args.out)

    rows = []
    with open(args.csv, 'r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    random.seed(args.seed)
    if args.sample is None and args.per_emotion is None:
        # write all
        selected = rows
    else:
        # sample in proportion to original distribution but cap per_emotion if set
        by_label = {}
        for r in rows:
            lbl = r['emotion']
            by_label.setdefault(lbl, []).append(r)
        selected = []
        if args.per_emotion is not None:
            for lbl, items in by_label.items():
                k = min(args.per_emotion, len(items))
                selected += random.sample(items, k)
        else:
            # args.sample total, proportional to counts
            total = args.sample
            counts = {lbl: len(items) for lbl, items in by_label.items()}
            total_rows = sum(counts.values())
            for lbl, items in by_label.items():
                k = max(1, int(round(counts[lbl] / total_rows * total)))
                k = min(k, len(items))
                selected += random.sample(items, k)
    # write images
    counters = {v:0 for v in FER_MAP.values()}
    for i, r in enumerate(selected):
        lbl = r['emotion']
        name = FER_MAP.get(lbl, 'unknown')
        try:
            img = pixels_to_image(r['pixels'])
        except Exception as e:
            print('Skipping row', i, 'due to', e)
            continue
        # save as PNG
        base = os.path.join(args.out, name)
        fname = f'{name}_{counters[name]:06d}.png'
        img.save(os.path.join(base, fname))
        counters[name] += 1
    print('Wrote images to', args.out)
    print('Counts:', counters)

if __name__ == '__main__':
    main()
