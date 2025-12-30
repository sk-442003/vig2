"""Generate a tiny synthetic audio dataset (sine waves) for demo/training purposes.
Creates WAV files under data/audio/example/<label>/ and writes a CSV at data/audio/example_audio_dataset.csv

Usage:
    python scripts/generate_synthetic_audio_dataset.py --out data/audio/example_audio_dataset.csv
"""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import os

LABELS = {
    'happy': [880, 1000],   # higher pitches
    'sad': [220, 260],      # lower pitches
    'neutral': [440, 440],  # middle
}

DURATION = 1.5  # seconds
SR = 22050


def generate_tone(freq, duration=DURATION, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), False)
    waveform = 0.5 * np.sin(2 * np.pi * freq * t)
    # add small noise
    waveform += 0.02 * np.random.randn(len(waveform))
    return waveform.astype(np.float32)


def main(out_csv, root='data/audio/example', n_per_label=None, counts=None, seed=42):
    """Create synthetic audio files.
    - If `counts` is provided (dict label->count), it is used per-label.
    - Else `n_per_label` is used for all labels.
    """
    np.random.seed(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    if counts is None:
        if n_per_label is None:
            n_per_label = 5
        counts = {label: n_per_label for label in LABELS.keys()}
    for label, freqs in LABELS.items():
        lab_dir = root / label
        lab_dir.mkdir(parents=True, exist_ok=True)
        cnt = counts.get(label, 0)
        for i in range(1, cnt+1):
            f = np.random.choice(freqs)
            w = generate_tone(f)
            fname = lab_dir / f"{label}_{i:04d}.wav"
            sf.write(str(fname), w, SR)
            rows.append((str(fname), label))
    # write CSV
    import csv
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['filepath','label'])
        w.writerows(rows)
    print(f"Created {len(rows)} synthetic WAV files in {root} and wrote CSV to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/audio/example_audio_dataset.csv')
    parser.add_argument('--n', type=int, default=None, help='Number of files per label (mutually exclusive with --total)')
    parser.add_argument('--total', type=int, default=None, help='Total number of files to generate across all labels')
    parser.add_argument('--root', default='data/audio/example', help='Root output directory')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.total is not None and args.n is not None:
        raise SystemExit('Specify either --n or --total, not both.')
    labels = list(LABELS.keys())
    if args.total is not None:
        # distribute total across labels as evenly as possible
        n_per_label = args.total // len(labels)
        remainder = args.total % len(labels)
        counts = {label: n_per_label + (1 if i < remainder else 0) for i, label in enumerate(labels)}
    else:
        n = args.n if args.n is not None else 5
        counts = {label: n for label in labels}
    try:
        main(args.out, root=args.root, counts=counts, seed=args.seed)
    except Exception as e:
        print('Failed to generate synthetic dataset:', e)
        raise
