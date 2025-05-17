import io
import tempfile
import requests
import gradio as gr
from fastai.vision.all import *
import matplotlib.pyplot as plt

def load_learner_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
        f.write(response.content)
        f.flush()
        learner = load_learner(f.name)
    return learner

model_urls = {
    "resnet34_base": "https://huggingface.co/MADemiral/garbage_prediction/resolve/main/resnet34_base.pkl",
    "resnet34_freeze_unfreeze_lr": "https://huggingface.co/MADemiral/garbage_prediction/resolve/main/resnet34_freeze_unfreeze_lr.pkl",
    "resnet50": "https://huggingface.co/MADemiral/garbage_prediction/resolve/main/resnet50.pkl"
}

learn1 = load_learner_from_url(model_urls["resnet34_base"])
learn2 = load_learner_from_url(model_urls["resnet34_freeze_unfreeze_lr"])
learn3 = load_learner_from_url(model_urls["resnet50"])

models = [
    ("Resnet34_base", learn1),
    ("Resnet34_freeze_unfreeze_lr", learn2),
    ("Resnet50", learn3)
]

def predict_and_plot(model, img):
    pred, idx, probs = model.predict(PILImage.create(img))
    categories = model.dls.vocab
    top3 = sorted(zip(categories, probs), key=lambda x: x[1], reverse=True)[:3]

    labels, values = zip(*top3)
    fig, ax = plt.subplots()
    ax.barh(labels, values, color="skyblue")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.invert_yaxis()
    ax.set_title("Top 3 Predictions")
    plt.tight_layout()

    return pred, fig

def classify_all_models(img):
    predictions = []
    plots = []
    for name, model in models:
        pred, fig = predict_and_plot(model, img)
        predictions.append(f"{name}: {pred}")
        plots.append(fig)
    return "\n".join(predictions), *plots

with gr.Blocks(title="Garbage Classification") as demo:
    gr.Markdown("## Garbage Classifier with 3 Models - Upload an Image")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image", type="filepath")
            gr.Examples(
                examples=["example_images/img1.jpg", "example_images/img2.jpg", "example_images/img3.jpg"],
                inputs=image_input,
                label="Examples",
            )
            predict_btn = gr.Button("Classify")

        with gr.Column():
            label_output = gr.Textbox(label="Predictions from All Models", lines=3)
            with gr.Row():
                plot1 = gr.Plot(label="Model Resnet34 Base Predictions")
                plot2 = gr.Plot(label="Model Resnet34 Freeze-Unfreeze Predictions")
                plot3 = gr.Plot(label="Model Resnet50 Predictions")

    predict_btn.click(fn=classify_all_models, inputs=image_input, outputs=[label_output, plot1, plot2, plot3])

demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
