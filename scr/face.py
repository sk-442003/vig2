from typing import Dict, Any
import numpy as np
try:
    from fer import FER
    _HAVE_FER = True
except Exception:
    FER = None
    _HAVE_FER = False
try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    cv2 = None
    _HAVE_CV2 = False

_detector = None

def _get_detector():
    global _detector
    if not _HAVE_FER:
        return None
    if _detector is None:
        _detector = FER(mtcnn=True)
    return _detector

def predict_face_from_array(img: np.ndarray) -> Dict[str, float]:
    """Predict emotion probabilities from an image array (BGR or RGB).

    Returns a dict mapping emotion labels to probabilities.
    """
    det = _get_detector()
    # ensure RGB
    if img.shape[-1] == 3:
        # fer expects RGB. Prefer cv2 if available; otherwise flip channels
        if _HAVE_CV2:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = img[..., ::-1]
    else:
        rgb = img
    if det is None:
        # fer not installed; return neutral-only result
        return {"angry":0.0,"disgust":0.0,"fear":0.0,"happy":0.0,"sad":0.0,"surprise":0.0,"neutral":1.0}

    results = det.detect_emotions(rgb)
    if not results:
        # no face found -> return neutral
        return {"angry":0.0,"disgust":0.0,"fear":0.0,"happy":0.0,"sad":0.0,"surprise":0.0,"neutral":1.0}
    # take the first face
    emotions = results[0].get("emotions", {})
    # normalize
    s = sum(emotions.values())
    if s == 0:
        return {k: 0.0 for k in emotions}
    return {k: float(v / s) for k, v in emotions.items()}
