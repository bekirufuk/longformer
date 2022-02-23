from ntpath import join
import os
import random
import json
import config
import pandas as pd
from datetime import date

import torch

import datasets
from datasets import load_dataset, load_metric

from IPython.display import display, HTML
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, LongformerConfig, TrainingArguments


def data_prep(batch, tokenizer):
    inputs = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=4096,
    )
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    #batch['attention_mask'][0] = [1 for i in range(len(batch['input_ids'][0]))]

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]
    batch["labels"] = batch["label"]
    return batch


if __name__ == '__main__':

    root_dir = os.path.expanduser("experiments")
    data_dir = os.path.expanduser("data/patentsview/example")
    model_dir = os.path.expanduser("models")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sanity_test = True
    model_prefix = 'longformer_sudo_finetuned_'

    # Load the data in two splits
    dataset = datasets.load_dataset('csv', data_files={'train': os.path.join(data_dir, 'train.csv'), 'test': os.path.join(data_dir, 'test.csv')},)

    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
    model     = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=config.num_labels)
    
    model.config = LongformerConfig.from_json_file(os.path.join(root_dir, 'longformer_config.json'))

    # If sanity_test is True only obtain a small fraction of the data to check if the pipeline is working.
    if sanity_test:
        train_dataset = dataset['train'].select(range(16))
        test_dataset  = dataset['test'].select(range(8))
    else:
        train_dataset = dataset['train']
        test_dataset  = dataset['test']
    
    # Tokenize and determine the attention masking for the inputs
    train_dataset = train_dataset.map(data_prep, batched=True, batch_size=config.batch_size, fn_kwargs={'tokenizer':tokenizer})
    test_dataset  = test_dataset.map(data_prep, batched=True, batch_size=config.batch_size, fn_kwargs={'tokenizer':tokenizer}) # remove_columns=column_list

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
        )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
        )

    # Define the trainer
    train_args = config.CustomTrainingArguments
    train_args.run_name = train_args.run_name + str(date.today())
    if device == 'cuda':
        train_args.fp16 = True

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=config.CustomTrainingArguments,
        compute_metrics=config.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    print('Training Complete')
    trainer.save_model(os.join(model_dir,model_prefix+str(date.today())))
    print('Model Saved under models/ with name: '+ model_prefix+str(date.today()))