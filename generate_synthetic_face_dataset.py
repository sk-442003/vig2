"""Generate a synthetic face-like dataset for quick experiments.
Creates images under data/face/example/<label>/ and can distribute a total number across labels.

Usage:
    python scripts/generate_synthetic_face_dataset.py --out data/face --total 900
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import random

LABELS = ["happy", "sad", "neutral"]
IMG_SIZE = (48, 48)


def draw_smiley(draw, bbox, mouth='smile'):
    left, top, right, bottom = bbox
    # eyes
    ex = left + (right - left) * 0.33
    ey = top + (bottom - top) * 0.35
    r = 2
    draw.ellipse((ex-r, ey-r, ex+r, ey+r), fill=0)
    ex = left + (right - left) * 0.67
    draw.ellipse((ex-r, ey-r, ex+r, ey+r), fill=0)
    # mouth
    mx1 = left + (right-left)*0.2
    mx2 = left + (right-left)*0.8
    my = top + (bottom-top)*0.7
    if mouth == 'smile':
        draw.arc((mx1, my-6, mx2, my+6), start=200, end=340, fill=0, width=2)
    elif mouth == 'frown':
        draw.arc((mx1, my-2, mx2, my+10), start=20, end=160, fill=0, width=2)
    else:
        draw.line((mx1, my, mx2, my), fill=0, width=2)


def create_face_image(label):
    img = Image.new('L', IMG_SIZE, color=255)  # white background
    draw = ImageDraw.Draw(img)
    w, h = IMG_SIZE
    # face circle
    margin = 4
    draw.ellipse((margin, margin, w-margin, h-margin), outline=0, width=1)
    if label == 'happy':
        draw_smiley(draw, (margin, margin, w-margin, h-margin), mouth='smile')
    elif label == 'sad':
        draw_smiley(draw, (margin, margin, w-margin, h-margin), mouth='frown')
    else:
        draw_smiley(draw, (margin, margin, w-margin, h-margin), mouth='neutral')
    # slight noise (random dots)
    for _ in range(random.randint(0, 8)):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        draw.point((x,y), fill=random.randint(0,80))
    return img


def main(out_dir, total=300, seed=42):
    random.seed(seed)
    out = Path(out_dir)
    labels = LABELS
    n_labels = len(labels)
    n_per = total // n_labels
    rem = total % n_labels
    counts = {l: n_per + (1 if i < rem else 0) for i,l in enumerate(labels)}
    for label, cnt in counts.items():
        lab_dir = out / label
        lab_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, cnt+1):
            img = create_face_image(label)
            fname = lab_dir / f"{label}_{i:04d}.png"
            img.save(fname)
    print(f"Created {total} synthetic face images under {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/face/example', help='Output directory root')
    parser.add_argument('--total', type=int, default=300, help='Total images to create (distributed across labels)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.out, args.total, args.seed)
