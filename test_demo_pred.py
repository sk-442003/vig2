"""Smoke test the multimodal predict_all function using small samples from data/ and print/save the result."""
import os
import json
from PIL import Image
from src.app import predict_all

# find sample face image
face_dir = 'data/face/example'
img_path = None
for root, dirs, files in os.walk(face_dir):
    for f in files:
        if f.lower().endswith(('.png','.jpg','.jpeg')):
            img_path = os.path.join(root, f)
            break
    if img_path:
        break

# find sample audio
aud_dir = 'data/audio/example'
audio_path = None
for root, dirs, files in os.walk(aud_dir):
    for f in files:
        if f.lower().endswith('.wav'):
            audio_path = os.path.join(root, f)
            break
    if audio_path:
        break

# sample text
text = 'I am feeling happy today.'
text_file = 'data/text/test.csv'
if os.path.exists(text_file):
    import pandas as pd
    df = pd.read_csv(text_file)
    if len(df) > 0:
        text = df['text'].iloc[0]

print('Using samples -> image:', img_path, 'audio:', audio_path, 'text sample:', text[:60])

img = Image.open(img_path) if img_path else None
audio = open(audio_path, 'rb') if audio_path else None

res = predict_all(img, audio, text)
print('Prediction result:')
print(json.dumps(res, indent=2))

os.makedirs('models/eval_reports', exist_ok=True)
open('models/eval_reports/demo_prediction.json','w',encoding='utf-8').write(json.dumps(res, indent=2))
print('Saved models/eval_reports/demo_prediction.json')

if audio:
    audio.close()