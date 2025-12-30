# scripts/evaluate_models.py
import argparse, json, os
import pandas as pd
from sklearn.metrics import classification_report
import joblib
import torch
from src.face import predict_face_from_array
from transformers import pipeline

def eval_audio_rf(model_path, csv):
    df = pd.read_csv(csv)
    clf = joblib.load(model_path)
    X = []
    # reuse extract_features from src.audio if available
    from src.audio import extract_features
    for fp in df["filepath"]:
        X.append(extract_features(fp))
    preds = clf.predict(X)
    print("Audio RF report:")
    print(classification_report(df["label"], preds))

def eval_text_model(model_dir, csv):
    df = pd.read_csv(csv)
    nlp = pipeline("text-classification", model=model_dir, return_all_scores=False)
    preds=[]
    for t in df["text"].tolist():
        out = nlp(t)
        # out is label string like 'LABEL_0' or {'label':"happy", 'score':..}
        if isinstance(out, list):
            label = out[0]["label"]
        else:
            label = out["label"]
        preds.append(label.lower())
    print("Text report:")
    print(classification_report(df["label"], preds))

def eval_face(model_path, data_dir):
    # expects ImageFolder val structure
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    import torch
    from torchvision import models
    dataset = ImageFolder(data_dir, transform=transforms.Compose([
        transforms.Resize((48,48)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(dataset.classes))
    ck = torch.load(model_path, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    preds=[]; trues=[]
    with torch.no_grad():
        for X,y in loader:
            X = X.to(device)
            out = model(X)
            p = torch.argmax(out, dim=1).cpu().numpy().tolist()
            preds.extend(p); trues.extend(y.numpy().tolist())
    print("Face report:")
    print(classification_report(trues, preds, target_names=dataset.classes))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_csv")
    parser.add_argument("--audio_model")
    parser.add_argument("--text_csv")
    parser.add_argument("--text_model")
    parser.add_argument("--face_dir")
    parser.add_argument("--face_model")
    args = parser.parse_args()

    if args.audio_csv and args.audio_model:
        eval_audio_rf(args.audio_model, args.audio_csv)
    if args.text_csv and args.text_model:
        eval_text_model(args.text_model, args.text_csv)
    if args.face_dir and args.face_model:
        eval_face(args.face_model, args.face_dir)