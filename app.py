import gradio as gr
from fastai.vision.all import *
import matplotlib.pyplot as plt

# Load multiple trained models
learn1 = load_learner('models/model1.pkl')
learn2 = load_learner('models/model2.pkl')
learn3 = load_learner('models/model3.pkl')

models = [
    ("Model 1", learn1),
    ("Model 2", learn2),
    ("Model 3", learn3)
]

# Predict function for one model
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

# Combined function for all models
def classify_all_models(img):
    predictions = []
    plots = []
    for name, model in models:
        pred, fig = predict_and_plot(model, img)
        predictions.append(f"{name}: {pred}")
        plots.append(fig)
    return predictions, plots

# Gradio Blocks UI
with gr.Blocks() as demo:
    gr.Markdown("## 🗑️ Garbage Classifier with 3 Models - Upload an Image")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image", type="filepath")
            predict_btn = gr.Button("Classify")

        with gr.Column():
            label_output = gr.Textbox(label="Predictions from All Models", lines=3)
            with gr.Row():
                plot1 = gr.Plot(label="Model 1 Predictions")
                plot2 = gr.Plot(label="Model 2 Predictions")
                plot3 = gr.Plot(label="Model 3 Predictions")

    predict_btn.click(fn=classify_all_models, inputs=image_input, outputs=[label_output, [plot1, plot2, plot3]])

demo.launch()
