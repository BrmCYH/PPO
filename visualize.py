import gradio as gr
def itm(item):
    strint = 'break'
    return item


def greet(name, intensity, item):
    x = itm(item)
    return "Hello," + name + "!"  * int(intensity)+","+x
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            projname = gr.Textbox(label="Project name")
            epochs = gr.Textbox(label= "personlist mark")
            ix = gr.Slider(label="train epoch")
        with gr.Column():
            translat_bth = gr.Button(value = "click")
    with gr.Row():
        name = gr.Text(label = "name")
    btn = gr.Button('Genearte')
    btn.click(greet, inputs = [projname, ix, epochs], outputs = [name])
    gr.Examples(["Mycyh"], inputs=[projname])# inputs take the value into projname block
demo.launch()

