
# Garbage Classification

This project implements an image classification system for garbage waste categorization using deep learning techniques. The primary objective is to classify images of garbage into predefined categories, enabling smarter waste management and recycling systems. The application supports multiple trained models and provides a visual interface for prediction and result interpretation.

## Project Overview

The `garbage_classification` system is built with the FastAI deep learning library and provides an interactive user interface using Gradio. The system allows users to upload an image of garbage, and then classifies it using multiple models trained on different data splits or architectures. The top three predicted categories are visualized using probability bar plots to offer interpretability.

## Dataset Structure

The training dataset is organized into class-labeled directories. Each subdirectory contains images that correspond to a particular garbage type:

```
dataset/
└── garbage_classification/
    ├── battery/
    ├── biological/
    ├── brown-glass/
    ├── cardboard/
    ├── clothes/
    ├── green-glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    ├── shoes/
    ├── trash/
    └── white-glass/
```

Each folder contains `.jpg` or `.png` images corresponding to the class label.

## Models

The project includes multiple trained models stored in the `models/` directory:

* `model1.pkl`
* `model2.pkl`
* `model3.pkl`

These models are trained using FastAI's `cnn_learner` on a dataset of garbage images. Each model may differ in terms of architecture, training data subset, or hyperparameters.

## Features

* Classification of garbage images into multiple categories
* Top-3 class prediction with probability scores
* Visualization of model predictions using bar plots
* Support for multiple model inference simultaneously

## Usage

### Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Ensure the `models/` directory contains the trained `.pkl` files.

### Running the Application

To launch the Gradio interface, run:

```bash
python app.py
```

A browser window will open with an image uploader. Upload an image to see predictions from all three models and their confidence scores.

## File Structure

```
.
├── app.py                  # Main Gradio interface with model loading and prediction
├── models/                 # Directory containing trained model .pkl files
├── dataset/                # Training dataset organized by class labels
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Acknowledgements

This project uses the FastAI library for model training and inference, and Gradio for building the user interface. The dataset used for training was manually curated or sourced from publicly available repositories for academic and research purposes.

## License

This project is intended for academic and research use only. Contact the author for licensing details if needed.

