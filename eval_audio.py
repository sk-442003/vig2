"""Evaluate audio RandomForest model on test.csv and save report."""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.audio import extract_features
import joblib, pandas as pd
from sklearn.metrics import classification_report
import os
os.makedirs('models/eval_reports', exist_ok=True)
print('Running audio evaluation...')
df = pd.read_csv('data/audio/test.csv')
clf = joblib.load('models/audio_rf_example_1000.joblib')
X = [extract_features(p) for p in df['filepath']]
preds = clf.predict(X)
report = classification_report(df['label'], preds)
print(report)
open('models/eval_reports/audio_report.txt','w',encoding='utf-8').write(report)
print('Saved models/eval_reports/audio_report.txt')