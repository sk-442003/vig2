"""Create augmented copies of audio, face images and text samples for simple class-balanced augmentation.
Usage examples:
  python scripts/augment_datasets.py --audio --n 2
  python scripts/augment_datasets.py --face --n 3
  python scripts/augment_datasets.py --text --n 2
"""
import argparse
import os
from src.augment import augment_audio_file, augment_image_file, augment_text_sample
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--audio', action='store_true')
parser.add_argument('--face', action='store_true')
parser.add_argument('--text', action='store_true')
parser.add_argument('--n', type=int, default=2, help='augmentations per sample')
args = parser.parse_args()

if args.audio:
    in_csv = 'data/audio/example_audio_dataset.csv'
    out_dir = 'data/audio/augmented'
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(in_csv)
    rows = []
    for idx, r in df.iterrows():
        base = r['filepath']
        lab = r['label']
        for i in range(args.n):
            out_path = os.path.join(out_dir, f"{lab}_{idx:04d}_aug{i}.wav")
            try:
                augment_audio_file(base, out_path)
                rows.append({'filepath': out_path, 'label': lab})
            except Exception as e:
                print('Skipping', base, '->', e)
    if rows:
        import csv
        out_csv = os.path.join(out_dir, 'augmented_audio_dataset.csv')
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print('Wrote', out_csv)

if args.face:
    in_dir = 'data/face/example'
    out_dir = 'data/face/augmented'
    os.makedirs(out_dir, exist_ok=True)
    # find images per class
    for root, dirs, files in os.walk(in_dir):
        for f in files:
            if f.lower().endswith(('.png','.jpg','.jpeg')):
                class_dir = os.path.basename(root)
                src = os.path.join(root, f)
                for i in range(args.n):
                    dst_dir = os.path.join(out_dir, class_dir)
                    os.makedirs(dst_dir, exist_ok=True)
                    dst = os.path.join(dst_dir, f.replace('.png', f'_aug{i}.png'))
                    try:
                        augment_image_file(src, dst)
                    except Exception as e:
                        print('Skipping', src, '->', e)
    print('Face augmentation done.')

if args.text:
    in_csv = 'data/text/train.csv'
    out_csv = 'data/text/augmented_train.csv'
    if not os.path.exists(in_csv):
        raise SystemExit('Text train file not found: '+in_csv)
    df = pd.read_csv(in_csv)
    rows = []
    for _, r in df.iterrows():
        text = r['text']
        lab = r['label']
        rows.append({'text': text, 'label': lab})
        for i in range(args.n):
            rows.append({'text': augment_text_sample(text), 'label': lab})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print('Wrote', out_csv)