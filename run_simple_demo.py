"""Run a minimal Gradio Interface that calls `src.app.predict_all`.
This avoids complex Blocks issues and should reliably start a local demo.
"""
import os, sys, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.app import predict_all
import gradio as gr

def fn(image, audio, text):
    res = predict_all(image, audio, text)
    return json.dumps(res, indent=2)

print('run_simple_demo: about to construct Interface')
try:
    iface = gr.Interface(fn=fn,
                         inputs=[gr.Image(type='pil', label='Face Image (optional)'), gr.Audio(type='filepath', label='Audio (optional)'), gr.Textbox(lines=3, label='Text (optional)')],
                         outputs=gr.Textbox(label='Results'))
    print('run_simple_demo: Interface constructed')
except Exception as e:
    import traceback
    print('Failed to construct Interface:', type(e), e)
    traceback.print_exc()
    raise

if __name__ == '__main__':
    print('Starting simple demo (allowing Gradio to pick an available port)')
    try:
        app, local_url, share_url = iface.launch(share=False, server_name='127.0.0.1', server_port=None, debug=True)
        print('Launched simple demo. local:', local_url, 'share:', share_url)
    except Exception as e:
        import traceback
        print('Failed to launch simple demo:', type(e), e)
        traceback.print_exc()
