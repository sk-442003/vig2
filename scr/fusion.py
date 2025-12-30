from typing import Dict, List

DEFAULT_LABELS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def align_probs(prob_dict: Dict[str, float], labels: List[str]=DEFAULT_LABELS) -> Dict[str, float]:
    # map incoming keys to expected labels; lowercase keys
    out = {l: 0.0 for l in labels}
    for k, v in prob_dict.items():
        key = k.lower()
        if key in out:
            out[key] = float(v)
    # normalize
    s = sum(out.values())
    if s > 0:
        return {k: float(v / s) for k, v in out.items()}
    return out

def fuse_modalities(modal_probs: List[Dict[str, float]], weights: List[float]=None) -> Dict[str, float]:
    """Fuse a list of modality probability dicts into a single probability distribution.

    `modal_probs` is list of dicts mapping label->prob.
    """
    n = len(modal_probs)
    if n == 0:
        return {}
    if weights is None:
        weights = [1.0] * n
    # align and weighted sum
    acc = {l:0.0 for l in DEFAULT_LABELS}
    for probs, w in zip(modal_probs, weights):
        p = align_probs(probs)
        for k in acc:
            acc[k] += p.get(k, 0.0) * w
    # normalize
    s = sum(acc.values())
    if s == 0:
        return acc
    return {k: float(v / s) for k, v in acc.items()}


# --- Meta-fusion support (trained meta-classifier) ---
import os
import joblib
import json
import numpy as np

def predict_meta(modal_probs_dict: Dict[str, Dict[str, float]],
                 meta_model_path: str = 'models/fusion_meta_lr.joblib',
                 labelmap_path: str = 'models/fusion_meta_labelmap.json') -> Dict[str, float] | None:
    """If a trained fusion meta-classifier is available, use it to predict final probabilities.

    Expects `modal_probs_dict` to be a mapping with keys 'audio','text','face' (any can be missing).
    Returns a dict label->prob if meta model found, otherwise None.
    """
    if not os.path.exists(meta_model_path) or not os.path.exists(labelmap_path):
        return None
    try:
        clf = joblib.load(meta_model_path)
        lm = json.load(open(labelmap_path, 'r', encoding='utf-8'))
        meta_classes = lm.get('classes') or []
        if not meta_classes:
            return None
        # build feature vector in training order: [audio_probs, text_probs, face_probs]
        audio_p = modal_probs_dict.get('audio', {})
        text_p = modal_probs_dict.get('text', {})
        face_p = modal_probs_dict.get('face', {})
        audio_vec = [float(audio_p.get(c, 0.0)) for c in meta_classes]
        text_vec = [float(text_p.get(c, 0.0)) for c in meta_classes]
        face_vec = [float(face_p.get(c, 0.0)) for c in meta_classes]
        feat = np.concatenate([audio_vec, text_vec, face_vec]).reshape(1, -1)
        probs = clf.predict_proba(feat)[0]
        # map the output probabilities to label names using the labelmap
        out = {str(lbl): float(probs[i]) if i < len(probs) else 0.0 for i, lbl in enumerate(meta_classes)}
        return out
    except Exception:
        return None
