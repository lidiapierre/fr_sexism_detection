from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset, ClassLabel
from huggingface_hub import login
import evaluate
import numpy as np
import os

import config
from data import post_process

from dotenv import load_dotenv
load_dotenv()

checkpoint = config.base_model

label2id = {"sexist": 1, "not sexist": 0}
id2label = {0: "not sexist", 1: "sexist"}

config = AutoConfig.from_pretrained(checkpoint, label2id=label2id, id2label=id2label)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint=checkpoint,
    num_labels=2,
    config=config
)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

metric = evaluate.load("glue", "mrpc")


def tokenize_function(example):
    return tokenizer(example["fr_sentences"], truncation=True)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    login(os.getenv('HUGGINGFACE_TOKEN'))
    # dataset = load_dataset("lidiapierre/fr_sexism_labelled")
    dataset = load_dataset("csv", data_files=config.fr_dataset)
    dataset = post_process(dataset)
    raw_dataset = dataset.train_test_split(test_size=0.1)
    tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # trainer.push_to_hub()


if __name__ == '__main__':
    main()

