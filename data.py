import pandas as pd
from transformers import pipeline
from datasets import ClassLabel
import config

en_fr_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")


def translate(text):
    return en_fr_translator(text)[0]['translation_text']


def create_fr_dataset():
    df = pd.read_excel(config.en_dataset)
    df["fr_sentences"] = df["Sentences"].map(translate)
    df.to_csv(config.fr_dataset, encoding='utf-8')


def adjust_labels(batch):
    batch["labels"] = [sentiment for sentiment in batch["Label"]]
    return batch


def post_process(dataset):
    features = dataset.features.copy()
    features["labels"] = ClassLabel(names=["not_sexist", "sexist"])
    dataset = dataset.map(adjust_labels, batched=True, features=features)
    dataset = dataset.remove_columns("Sentences")
    dataset = dataset.remove_columns("Label")
    return dataset
