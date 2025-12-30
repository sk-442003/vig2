"""Evaluate a PyTorch audio CNN model on data/audio/test.csv and save a report.
Saves report to models/eval_reports/audio_cnn_small_report.txt and a confusion matrix PNG if matplotlib available.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch import nn

# Minimal dataset loader (same preprocessing as training)
import librosa

class EvalAudioDataset(torch.utils.data.Dataset):
    def __init__(self, filepaths, labels, sr=22050, n_mels=64, duration=3.0):
        self.filepaths = filepaths
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.samples = int(sr * duration)
    def __len__(self):
        return len(self.filepaths)
    def __getitem__(self, i):
        fp = self.filepaths[i]
        y, _ = librosa.load(fp, sr=self.sr)
        if len(y) < self.samples:
            y = np.pad(y, (0, self.samples - len(y)))
        else:
            y = y[:self.samples]
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = np.expand_dims(mel_db, 0).astype(np.float32)  # 1 x n_mels x time
        return mel_db, self.labels[i]

class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(), nn.Linear(64, n_classes)
        )
    def forward(self,x): return self.net(x)


def main():
    os.makedirs('models/eval_reports', exist_ok=True)
    test_csv = 'data/audio/test.csv'
    if not os.path.exists(test_csv):
        raise SystemExit('Test CSV not found: '+test_csv)
    df = pd.read_csv(test_csv)
    filepaths = df['filepath'].tolist()
    trues = df['label'].tolist()

    # load model
    model_path = 'models/audio_cnn_small.pth'
    if not os.path.exists(model_path):
        raise SystemExit('Model not found: '+model_path)
    try:
        ck = torch.load(model_path, map_location='cpu')
    except Exception as e:
        # Fallback: allow weights_only=False (may be required for older checkpoints)
        try:
            ck = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            # weights_only not supported in this torch version; re-raise original
            raise
        except Exception:
            raise
    classes = ck.get('classes') or ck.get('labels') or ['label_'+str(i) for i in range(3)]
    model = SimpleCNN(n_classes=len(classes))
    # support older checkpoints that store only state_dict (or the dict wrapper)
    sd = ck.get('model_state_dict') if isinstance(ck, dict) and 'model_state_dict' in ck else ck
    model.load_state_dict(sd)
    model.eval()

    ds = EvalAudioDataset(filepaths, trues)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=0)

    preds=[]; y_true=[]
    with torch.no_grad():
        for X,y in loader:
            out = model(torch.from_numpy(np.array(X)))
            p = out.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(p); y_true.extend(y)

    # Map numeric preds to label names if possible
    pred_labels = [classes[p] if p < len(classes) else str(p) for p in preds]

    report = classification_report(y_true, pred_labels)
    cm = confusion_matrix(y_true, pred_labels, labels=sorted(list(set(y_true))))

    rpt_path = 'models/eval_reports/audio_cnn_small_report.txt'
    open(rpt_path,'w',encoding='utf-8').write(report)
    print('Saved', rpt_path)

    # try to save confusion matrix plot (use Agg backend for headless environments)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        labels_sorted = sorted(list(set(y_true)))
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels_sorted, yticklabels=labels_sorted)
        plt.ylabel('True'); plt.xlabel('Pred')
        img_path = 'models/eval_reports/audio_cnn_small_confusion.png'
        plt.savefig(img_path)
        plt.close()
        print('Saved', img_path)
    except Exception as e:
        import traceback
        print('Skipping confusion plot:', type(e), e)
        traceback.print_exc()

    print('--- classification report ---')
    print(report)

if __name__ == '__main__':
    main()
