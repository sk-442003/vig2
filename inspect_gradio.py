import gradio as gr
import inspect
print('gradio', gr.__version__)
print('Audio signature:', inspect.signature(gr.Audio))
print('\nAudio doc (first 10 lines):')
for i, line in enumerate(gr.Audio.__doc__.splitlines()[:10]):
    print(i+1, line)
