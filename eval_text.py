"""Evaluate text HuggingFace model on data/text/test.csv and save report."""
import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import classification_report

os.makedirs('models/eval_reports', exist_ok=True)
print('Running text evaluation...')

df = pd.read_csv('data/text/test.csv')
texts = df['text'].tolist()
labels = df['label'].tolist()
print('Loaded', len(texts), 'texts; sample:', texts[0][:80])

model_dir = 'models/text_roberta_example'
print('Loading tokenizer and model from', model_dir)
try:
    # The saved model dir may not include tokenizer files; load a compatible tokenizer instead
    base_tokenizer = 'distilroberta-base'
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
    print('Tokenizer loaded; vocab size:', getattr(tokenizer, 'vocab_size', 'n/a'))
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    print('Model loaded; num labels:', model.config.num_labels)
except Exception as e:
    print('Failed to load model/tokenizer:', e)
    raise

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(dev).eval()

preds = []
bs = 32
with torch.no_grad():
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(dev)
        out = model(**enc)
        p = out.logits.argmax(dim=1).cpu().numpy().tolist()
        preds.extend(p)

# map numeric preds back to label names if possible
label_map_path = os.path.join(model_dir, 'label_map.json')
if os.path.exists(label_map_path):
    import json
    lm = json.load(open(label_map_path))
    inv = {int(v):k for k,v in lm.items()}
    pred_labels = [inv.get(p,str(p)) for p in preds]
else:
    pred_labels = [str(p) for p in preds]

report = classification_report(labels, pred_labels)
print(report)
open('models/eval_reports/text_report.txt','w',encoding='utf-8').write(report)
print('Saved models/eval_reports/text_report.txt')