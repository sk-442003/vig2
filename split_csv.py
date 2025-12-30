# scripts/split_csv.py
import argparse
import csv
from pathlib import Path
import random

def main(csv_path, out_dir, test_size=0.2, seed=42):
    random.seed(seed)
    p = Path(csv_path)
    rows = []
    with open(p, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row['filepath'], row['label']))
    random.shuffle(rows)
    n = len(rows)
    n_test = int(n * test_size)
    test = rows[:n_test]
    train = rows[n_test:]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    def write(name, data):
        with open(out / name, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['filepath','label'])
            w.writerows(data)
    write('train.csv', train)
    write('test.csv', test)
    print(f"Wrote train={len(train)} test={len(test)} to {out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('--out', default='data/audio')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.csv, args.out, args.test_size, args.seed)
