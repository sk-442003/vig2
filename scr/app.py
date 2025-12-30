import numpy as np
import tempfile
import os
from typing import Dict
from PIL import Image

from .face import predict_face_from_array
from .text import predict_text_emotion
from .audio import predict_audio
from .fusion import fuse_modalities, predict_meta

# avoid TensorFlow auto-import side effects when using transformers/pipelines
os.environ['TRANSFORMERS_NO_TF'] = '1'
# Prefer the augmented audio RF model (if exists), fall back to example or generic RF
DEFAULT_AUDIO_MODEL = os.path.join(os.path.dirname(__file__), "../models/audio_rf_aug.joblib")
if not os.path.exists(DEFAULT_AUDIO_MODEL):
    DEFAULT_AUDIO_MODEL = os.path.join(os.path.dirname(__file__), "../models/audio_rf_example_1000.joblib")
if not os.path.exists(DEFAULT_AUDIO_MODEL):
    DEFAULT_AUDIO_MODEL = os.path.join(os.path.dirname(__file__), "../models/audio_rf.joblib")
print('Using audio model:', DEFAULT_AUDIO_MODEL)

def image_to_np(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    # ensure BGR for face module input; avoid requiring cv2 at import-time
    if arr.shape[-1] == 3:
        # flip RGB->BGR using numpy slicing
        return arr[..., ::-1]
    return arr

def predict_all(image, audio, text, audio_model_path=None):
    results = {}
    modal_probs = []
    if image is not None:
        img_arr = image_to_np(image)
        face_probs = predict_face_from_array(img_arr)
        results['face'] = face_probs
        modal_probs.append(face_probs)
    if audio is not None:
        # audio can be a file path (str), a file-like with .name, or a file-like with .read()
        audio_path = None
        if isinstance(audio, str) and os.path.exists(audio):
            audio_path = audio
        elif hasattr(audio, 'name'):
            audio_path = audio.name
        elif hasattr(audio, 'read'):
            # save to temp
            fd, audio_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            with open(audio_path, 'wb') as f:
                f.write(audio.read())
        if audio_path is None or not os.path.exists(audio_path):
            audio_probs = {"neutral": 1.0}
        else:
            model_path = audio_model_path or DEFAULT_AUDIO_MODEL
            try:
                audio_probs = predict_audio(model_path, audio_path)
            except Exception:
                audio_probs = {"neutral": 1.0}
        results['audio'] = audio_probs
        modal_probs.append(audio_probs)
    if text is not None and text.strip() != "":
        text_probs = predict_text_emotion(text)
        results['text'] = text_probs
        modal_probs.append(text_probs)
    # Prefer meta-classifier fusion when available (more robust than simple averaging)
    meta_pred = None
    try:
        meta_pred = predict_meta(results)
    except Exception:
        meta_pred = None
    if meta_pred is not None:
        fused = meta_pred
    else:
        fused = fuse_modalities(modal_probs)
    results['fused'] = fused
    # also return best label
    best = max(fused.items(), key=lambda x: x[1]) if fused else (None, 0.0)
    results['prediction'] = {"label": best[0], "score": best[1]}
    return results

def build_demo():
    print('build_demo: start')
    import gradio as gr
    print('build_demo: imported gradio', getattr(gr, '__version__', None))
    with gr.Blocks() as demo:
        print('build_demo: created Blocks')
        gr.Markdown("# Multimodal Emotion Recognition (Face, Audio, Text)")
        with gr.Row():
            with gr.Column():
                print('build_demo: creating image component')
                img_in = gr.Image(type="pil", label="Face Image (optional)")
                print('build_demo: creating audio component')
                # Create Audio component in a backward/forward compatible way across Gradio versions
                audio_in = None
                for aud_args in (
                    {"label": "Audio (optional)", "type": "filepath"},
                    {"type": "filepath", "label": "Audio (optional)"},
                    {"label": "Audio (optional)"},
                ):
                    try:
                        audio_in = gr.Audio(**aud_args)
                        break
                    except TypeError:
                        audio_in = None
                if audio_in is None:
                    # final fallback: construct with explicit filepath type
                    audio_in = gr.Audio(type='filepath', label="Audio (optional)")
                print('build_demo: creating text component')
                text_in = gr.Textbox(lines=3, placeholder="Enter text...", label="Text (optional)")
                print('build_demo: creating button')
                btn = gr.Button("Predict")
            with gr.Column():
                print('build_demo: creating output json')
                out_json = gr.JSON(label="Results")

        def run(image, audio, text):
            return predict_all(image, audio, text)

        print('build_demo: wiring button click')
        btn.click(run, inputs=[img_in, audio_in, text_in], outputs=[out_json])

    print('build_demo: returning demo')
    return demo

if __name__ == '__main__':
    demo = build_demo()
    demo.launch()
