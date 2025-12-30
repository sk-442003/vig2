# scripts/train_audio_cnn.py
import argparse, os, joblib
import numpy as np
import pandas as pd
import librosa
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

class AudioDataset(Dataset):
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

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total=0
    for X,y in loader:
        X = X.to(device); y = y.to(device)
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward(); opt.step()
        total += loss.item()
    return total/len(loader)

def eval_model(model, loader, device):
    model.eval()
    preds=[]; trues=[]
    with torch.no_grad():
        for X,y in loader:
            X = X.to(device)
            out = model(X)
            p = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(p.tolist()); trues.extend(y.numpy().tolist())
    return preds, trues

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV with filepath,label")
    parser.add_argument("--out", default="models/audio_cnn.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    filepaths = df["filepath"].tolist()
    labels = df["label"].tolist()
    le = LabelEncoder(); y = le.fit_transform(labels)
    X_train, X_val, y_train, y_val = train_test_split(filepaths, y, test_size=0.2, random_state=42, stratify=y)

    train_ds = AudioDataset(X_train, y_train)
    val_ds = AudioDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(n_classes=len(le.classes_)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, opt, loss_fn, device)
        preds, trues = eval_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} loss={train_loss:.4f}")
        print(classification_report(trues, preds, target_names=le.classes_, digits=4))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "classes": list(le.classes_)}, args.out)
    print("Saved audio CNN model to", args.out)

if __name__ == "__main__":
    main()