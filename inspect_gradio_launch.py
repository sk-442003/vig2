import gradio as gr
import inspect
print('gradio', gr.__version__)
print('Blocks.launch signature:')
print(inspect.signature(gr.Blocks.launch))
print('\nBlocks.launch doc (first 20 lines):')
print('\n'.join(gr.Blocks.launch.__doc__.splitlines()[:20]))
