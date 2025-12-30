# scripts/prepare_text_split.py
import argparse
from pathlib import Path
import csv
import random


def main(csv_in, out_dir, val_size=0.1, test_size=0.1, seed=42):
    rows = []
    with open(csv_in, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row['text'], row['label']))
    random.seed(seed)
    random.shuffle(rows)
    n = len(rows)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    test = rows[:n_test]
    val = rows[n_test:n_test+n_val]
    train = rows[n_test+n_val:]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    def write(name, data):
        with open(out / f"{name}.csv", 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['text','label'])
            w.writerows(data)
    write('train', train)
    write('val', val)
    write('test', test)
    print(f"Wrote train/val/test to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Input CSV with text,label")
    parser.add_argument("--out", default="data/text")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.csv, args.out, args.val_size, args.test_size, args.seed)
