"""Run the small Gradio demo using the RF audio model and the fusion meta-classifier if available.
Usage: python scripts/run_demo.py
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.app import build_demo

if __name__ == '__main__':
    demo = build_demo()
    # run locally on a fixed port and block (print any errors)
    try:
        print('Launching Gradio demo on http://127.0.0.1:7860 (debug=True)')
        app, local_url, share_url = demo.launch(share=False, server_name='127.0.0.1', server_port=7860, debug=True)
        print('Launched demo. local:', local_url, 'share:', share_url)
    except Exception as e:
        print('Failed to launch demo:', type(e), e)

