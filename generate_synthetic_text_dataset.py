# scripts/generate_synthetic_text_dataset.py
import argparse
import csv
from pathlib import Path
import random

LABELS = ['happy', 'sad', 'neutral']
EXAMPLES = {
    'happy': [
        "I am feeling great today!",
        "This is the best day ever.",
        "I'm so happy and excited.",
        "What a wonderful surprise!",
        "I love this!"
    ],
    'sad': [
        "I am feeling down.",
        "This is a bad day.",
        "I am so sad right now.",
        "I feel terrible and low.",
        "I don't feel good about this."
    ],
    'neutral': [
        "I went to the store.",
        "The meeting starts at 3pm.",
        "It is an ordinary day.",
        "He went home after work.",
        "It will rain tomorrow."
    ]
}


def main(out_dir='data/text', total=1200, val_size=0.1, test_size=0.1, seed=42, train_count=None, val_count=None, test_count=None):
    random.seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_labels = len(LABELS)

    # generate rows according to total (distributed across labels)
    n_per_label = total // n_labels
    remainder = total % n_labels
    rows = []
    for i, label in enumerate(LABELS):
        cnt = n_per_label + (1 if i < remainder else 0)
        for j in range(cnt):
            text = random.choice(EXAMPLES[label])
            # add small randomization
            text = text + " " + random.choice(["", "Really.", "I think so.", "Indeed."])
            rows.append((text, label))
    random.shuffle(rows)

    # If explicit counts provided, split accordingly
    if train_count is not None and val_count is not None and test_count is not None:
        if train_count + val_count + test_count != len(rows):
            # if rows length differs from requested total, regenerate rows to match sum
            total_req = train_count + val_count + test_count
            rows = []
            n_per_label = total_req // n_labels
            remainder = total_req % n_labels
            for i, label in enumerate(LABELS):
                cnt = n_per_label + (1 if i < remainder else 0)
                for j in range(cnt):
                    text = random.choice(EXAMPLES[label])
                    text = text + " " + random.choice(["", "Really.", "I think so.", "Indeed."])
                    rows.append((text, label))
            random.shuffle(rows)
        n = len(rows)
        train = rows[:train_count]
        val = rows[train_count:train_count+val_count]
        test = rows[train_count+val_count:train_count+val_count+test_count]
    else:
        n = len(rows)
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        test = rows[:n_test]
        val = rows[n_test:n_test+n_val]
        train = rows[n_test+n_val:]

    def write(name, data):
        with open(out / f"{name}.csv", 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['text','label'])
            w.writerows(data)
    write('train', train)
    write('val', val)
    write('test', test)
    print(f"Wrote {len(train)} train, {len(val)} val, {len(test)} test to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/text')
    parser.add_argument('--total', type=int, default=1200)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_count', type=int, default=None, help='Exact number of train samples')
    parser.add_argument('--val_count', type=int, default=None, help='Exact number of validation samples')
    parser.add_argument('--test_count', type=int, default=None, help='Exact number of test samples')
    args = parser.parse_args()
    main(args.out, args.total, args.val_size, args.test_size, args.seed, args.train_count, args.val_count, args.test_count)
