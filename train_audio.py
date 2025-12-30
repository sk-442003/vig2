"""Train audio classifier script.
Usage: prepare a CSV with `filepath,label` and run this script.
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import sys
import argparse
import pandas as pd
# make project root importable when running scripts directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.audio import train_audio_classifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV file with filepath,label columns")
    parser.add_argument("--out", default="models/audio_rf.joblib")
    args = parser.parse_args()

    # basic validation
    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV file not found: {args.csv}. Use scripts/prepare_audio_csv.py to generate one.")
    df = pd.read_csv(args.csv)
    if 'filepath' not in df.columns or 'label' not in df.columns:
        raise SystemExit("CSV must contain 'filepath' and 'label' columns")
    missing = [p for p in df['filepath'].tolist() if not os.path.exists(p)]
    if missing:
        example = missing[:5]
        raise SystemExit(f"{len(missing)} audio files referenced in CSV were not found. Examples: {example}")

    filepaths = df['filepath'].tolist()
    labels = df['label'].tolist()
    out = train_audio_classifier(filepaths, labels, args.out)
    print("Saved model to", out)

if __name__ == '__main__':
    main()
