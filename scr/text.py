import os
os.environ['TRANSFORMERS_NO_TF'] = '1'
from transformers import pipeline
from typing import Dict

_nlp = None

def _get_pipeline():
    global _nlp
    if _nlp is None:
        # prefer a local fine-tuned model when available
        local_model = 'models/text_roberta_example'
        if os.path.exists(local_model):
            # use local tokenizer + model to avoid downloads and protobuf/tokenizer issues
            _nlp = pipeline("text-classification", model=local_model, tokenizer='distilroberta-base', return_all_scores=True)
        else:
            # fall back to public HF model
            _nlp = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    return _nlp

def predict_text_emotion(text: str) -> Dict[str, float]:
    nlp = _get_pipeline()
    out = nlp(text)
    # pipeline returns list of lists
    if isinstance(out, list) and len(out) > 0:
        scores = out[0]
        total = sum([s["score"] for s in scores])
        if total == 0:
            return {s["label"]: 0.0 for s in scores}
        return {s["label"].lower(): float(s["score"]) for s in scores}
    return {}
