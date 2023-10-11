# FR_sexism_detection

![Python version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-transformers%20v4.6.1%2B-brightgreen)

This repository contains a script to fine-tune a pre-trained multilingual Transformers model for binary classification (sexist/not sexist) in French. The model is based on [DistilBERT](https://huggingface.co/distilbert-base-multilingual-cased) and has been fine-tuned on the [FR_sexism_labelled](https://huggingface.co/datasets/lidiapierre/fr_sexism_labelled) dataset.

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/FR_sexism_detection.git
   cd FR_sexism_detection
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Fine-tune the model by running the `model.py` script.

4. After fine-tuning, the model can be pushed to the HuggingFace hub.

## Pre-trained Model

- Pre-trained model: [distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased)

## Fine-Tuning Dataset

- Fine-tuning dataset: [FR_sexism_labelled](https://huggingface.co/datasets/lidiapierre/fr_sexism_labelled)

## Model

You can find my final fine-tuned model on Hugging Face Model Hub: [lidiapierre/distilbert-base-multi-fr-sexism](https://huggingface.co/lidiapierre/distilbert-base-multi-fr-sexism)

It achieves the following results on the evaluation set:
- Loss: 0.3751
- Accuracy: 0.9123
- F1: 0.9206

Feel free to try it out on my [demo hosted on Spaces](https://huggingface.co/spaces/lidiapierre/FR-sexism-detection)!
