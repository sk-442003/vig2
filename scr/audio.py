import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from typing import List, Dict

def extract_features(path: str, sr: int = 22050, n_mfcc: int = 13) -> np.ndarray:
    y, sr = librosa.load(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # aggregate stats
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return feat

def train_audio_classifier(filepaths: List[str], labels: List[str], out_path: str):
    X = [extract_features(p) for p in filepaths]
    X = np.stack(X)
    le_labels = labels
    X_train, X_test, y_train, y_test = train_test_split(X, le_labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    joblib.dump(clf, out_path)
    return out_path

def predict_audio(model_path: str, audio_path: str) -> Dict[str, float]:
    clf = joblib.load(model_path)
    feat = extract_features(audio_path)
    proba = clf.predict_proba([feat])[0]
    classes = clf.classes_
    return {str(c): float(p) for c, p in zip(classes, proba)}
