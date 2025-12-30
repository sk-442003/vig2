# scripts/prepare_audio_csv.py
import argparse
import csv
from pathlib import Path


def main(root_dir, out_csv, exts):
    files = []
    root = Path(root_dir)
    for p in root.rglob('*'):
        if p.suffix.lower() in exts and p.is_file():
            # label = parent folder name
            label = p.parent.name
            files.append((str(p.resolve()), label))
    files.sort()
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['filepath','label'])
        w.writerows(files)
    print(f"Wrote {len(files)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/audio", help="Root folder with subfolders per label")
    parser.add_argument("--out", default="data/audio/audio_dataset.csv", help="Output CSV path")
    parser.add_argument("--exts", default=".wav,.flac,.mp3", help="Comma-separated extensions")
    args = parser.parse_args()
    exts = [e.lower().strip() for e in args.exts.split(',')]
    main(args.root, args.out, exts)
