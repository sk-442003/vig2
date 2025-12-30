"""Merge original audio train.csv (if present) with augmented audio CSV into a single train file for CNN training."""
import pandas as pd
import os
orig = 'data/audio/train.csv'
aug = 'data/audio/augmented/augmented_audio_dataset.csv'
out = 'data/audio/train_augmented.csv'
frames = []
if os.path.exists(orig):
    frames.append(pd.read_csv(orig))
if os.path.exists(aug):
    frames.append(pd.read_csv(aug))
if not frames:
    raise SystemExit('No input CSVs found')
df = pd.concat(frames, ignore_index=True)
df.to_csv(out, index=False)
print('Wrote', out, 'rows=', len(df))