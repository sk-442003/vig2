# scripts/prepare_face_from_csv.py
import argparse
from pathlib import Path
import shutil
import random


def main(csv_path, out_dir, val_size=0.2, seed=42):
    import csv
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row['filepath'], row['label']))
    random.seed(seed)
    groups = {}
    for fp,label in rows:
        groups.setdefault(label, []).append(fp)
    out = Path(out_dir)
    for split in ('train','val'):
        for label in groups:
            (out / split / label).mkdir(parents=True, exist_ok=True)
    for label, files in groups.items():
        random.shuffle(files)
        n_val = max(1, int(len(files) * val_size))
        val_files = files[:n_val]
        train_files = files[n_val:]
        for f in train_files:
            shutil.copy2(f, out / 'train' / label / Path(f).name)
        for f in val_files:
            shutil.copy2(f, out / 'val' / label / Path(f).name)
    print(f"Prepared face folders at {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV with filepath,label for face images")
    parser.add_argument("--out", default="data/face")
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.csv, args.out, args.val_size, args.seed)
