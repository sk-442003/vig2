"""Train a meta-classifier (logistic regression) on modality predictions.
This script gathers predictions from current modality models on a dataset split
and trains a simple logistic regression that maps concatenated probabilities to labels.
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from src.audio import predict_audio
from src.face import predict_face_from_array
from src.text import predict_text_emotion

# expects: data/* test/val csvs and models present
AUDIO_TEST = 'data/audio/test.csv'
TEXT_TEST = 'data/text/test.csv'
FACE_DIR = 'data/face/example'

assert os.path.exists(AUDIO_TEST) and os.path.exists(TEXT_TEST) and os.path.exists(FACE_DIR), 'Required data or splits missing.'

# load dataframes
aud_df = pd.read_csv(AUDIO_TEST)
text_df = pd.read_csv(TEXT_TEST)

# align datasets by label counts; for simplicity, sample up to SAMPLE_N and use batched text inference
SAMPLE_N = 200
labels = sorted(list(set(aud_df['label']).union(set(text_df['label']))))

# prepare audio predictions (sampled)
print('Computing audio predictions...')
from joblib import load
AUDIO_MODEL = 'models/audio_rf_example_1000.joblib'
clf = load(AUDIO_MODEL)

X_probs = []
y = []
for _, row in aud_df.iloc[:SAMPLE_N].iterrows():
    p = predict_audio(AUDIO_MODEL, row['filepath'])
    # ensure consistent ordering of classes
    probs = [p.get(cls, 0.0) for cls in clf.classes_]
    X_probs.append(probs)
    y.append(row['label'])

# create text predictions using batched inference to avoid repeated pipeline overhead
print('Computing text predictions (batched)...')
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
TEXT_MODEL_DIR = 'models/text_roberta_example'
tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR, local_files_only=True)
model.eval()

def batch_text_probs(texts, batch_size=32):
    out = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits.cpu().numpy()
        # softmax
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        out.extend(probs.tolist())
    return out

texts = text_df['text'].iloc[:SAMPLE_N].tolist()
text_probs_raw = batch_text_probs(texts)
# map text probs to label order in `labels` set — if label_map exists, use it
import json
label_map_path = os.path.join(TEXT_MODEL_DIR, 'label_map.json')
if os.path.exists(label_map_path):
    lm = json.load(open(label_map_path))
    inv = {int(v):k for k,v in lm.items()}
    # model logits are ordered by label id
    text_probs_list = []
    for probs in text_probs_raw:
        mapped = [0.0]*len(labels)
        for idx, val in enumerate(probs):
            label_name = inv.get(idx, str(idx))
            if label_name in labels:
                mapped[labels.index(label_name)] = val
        text_probs_list.append(mapped)
else:
    # fallback: just allocate into labels length (best-effort)
    text_probs_list = [[0.0]*len(labels) for _ in text_probs_raw]

# create face predictions by scanning face dir and mapping labels (sampled)
print('Computing face predictions...')
face_files = []
for root, dirs, files in os.walk(FACE_DIR):
    for f in files:
        if f.lower().endswith(('.png','.jpg','.jpeg')):
            face_files.append(os.path.join(root,f))
face_files = sorted(face_files)[:SAMPLE_N]
face_probs_list = []
face_labels = []
for f in face_files:
    lab = os.path.basename(os.path.dirname(f))
    face_labels.append(lab)
    fp = predict_face_from_array(__import__('numpy').asarray(__import__('PIL').Image.open(f)))
    face_probs_list.append([fp.get(l,0.0) for l in labels])

# For meta training we need aligned X and y; for a simple start we'll take min length
min_n = min(len(X_probs), len(text_probs_list), len(face_probs_list))
if min_n == 0:
    raise RuntimeError('No data available to train fusion meta-classifier (one modality missing predictions).')

X = []
y = []
for i in range(min_n):
    features = np.concatenate([X_probs[i], text_probs_list[i], face_probs_list[i]])
    X.append(features)
    # prefer audio label for target (assuming consistent label assignment in synthetic dataset)
    y.append(aud_df['label'].iloc[i])

X = np.stack(X)
le = LabelEncoder()
y_enc = le.fit_transform(y)
print('Training meta-classifier on', X.shape)
clf_meta = LogisticRegression(max_iter=200)
clf_meta.fit(X, y_enc)

preds = clf_meta.predict(X)
print(classification_report(y_enc, preds, target_names=le.classes_))

# save meta model and label encoder mapping
import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(clf_meta, 'models/fusion_meta_lr.joblib')
joblib.dump({'classes': list(le.classes_)}, 'models/fusion_meta_labelmap.json')
print('Saved models/fusion_meta_lr.joblib and labelmap')