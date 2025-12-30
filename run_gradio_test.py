import gradio as gr

def f(x):
    return x

i = gr.Interface(fn=f, inputs=gr.Textbox(), outputs=gr.Textbox())
print('Launching small test app on 7860')
app, local_url, share_url = i.launch(share=False, server_name='127.0.0.1', server_port=7860, debug=True)
print('Launched', local_url, share_url)
