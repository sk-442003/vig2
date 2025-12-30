# scripts/validate_audio_dataset.py
import argparse
from pathlib import Path
import librosa


def main(csv_path, sample=3):
    missing = []
    unreadable = []
    rows = []
    import csv
    with open(csv_path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for i,row in enumerate(r):
            p = Path(row['filepath'])
            rows.append(row)
            if not p.exists():
                missing.append(str(p))
            elif i < sample:
                try:
                    _ = librosa.load(str(p), sr=22050, duration=1.0)
                except Exception as e:
                    unreadable.append((str(p), str(e)))
    print(f"Total rows: {len(rows)}")
    print(f"Missing files: {len(missing)}")
    if missing:
        print('\n'.join(missing[:20]))
    print(f"Unreadable samples: {len(unreadable)}")
    for u in unreadable:
        print(u)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_audio_dataset.py data/audio/audio_dataset.csv")
    else:
        main(sys.argv[1])
