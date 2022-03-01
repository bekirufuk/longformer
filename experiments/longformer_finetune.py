import os
from tkinter.filedialog import test
import config, utils
import pandas as pd

from tqdm.auto import tqdm

from accelerate import Accelerator

import torch
from torch.utils.data import DataLoader

from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, LongformerConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler

from datasets import Features, Value, ClassLabel, load_dataset, load_metric


def batch_tokenizer(batch):
    return tokenizer(batch["text"],
    padding='max_length',
    truncation=True
    )


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_dir = os.path.expanduser('data/patentsview/example')
root_dir = os.path.expanduser("experiments")
small_scale = True


if __name__ == '__main__':

    torch.cuda.empty_cache()

    accelerator = Accelerator(fp16=True)

    class_names = config.labels_list
    features = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})
    data_files = {"train":os.path.join(data_dir, 'train.csv'), "test":os.path.join(data_dir, 'test.csv')}

    dataset = load_dataset('csv', data_files=data_files, features=features)
    
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") # Vanilla Usage
    # tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384") # LED model might be used for 16.384 token long inputs.
    # tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096") # Longformer Alternative

    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length = config.max_length)

    tokenized_data = dataset.map(batch_tokenizer, batched=True, remove_columns=['text'])
    tokenized_data = tokenized_data.rename_column("label","labels")
    tokenized_data.set_format("torch")

    # tokenized_data.save_to_disk(os.path.join(data_dir, "tokenized_data"))

    if small_scale:
        train_data = tokenized_data["train"].shuffle(seed=config.seed).select(range(16))
        test_data = tokenized_data["test"].shuffle(seed=config.seed).select(range(8))
    else:
        train_data = tokenized_data["train"]
        test_data = tokenized_data["test"]

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=config.batch_size)


    longformer_config = LongformerConfig.from_json_file(os.path.join(root_dir, 'longformer_config.json'))
    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
        config=longformer_config
    )
    model.gradient_checkpointing_enable()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_dataloader, test_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, test_dataloader, model, optimizer
    )
    
    num_training_steps = config.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    progress_bar = tqdm(range(num_training_steps))
    
    global_attention_mask = utils.attention_mapper(device)

    model.train()
    print("\n----------\n TRAINING STARTED \n----------\n")
    for epoch in range(config.num_epochs):
        for batch in train_dataloader:
            outputs = model(**batch, global_attention_mask=global_attention_mask)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    accelerator.free_memory()

    print("Training Completed")
    print("Evaluation Started")
    model.eval()
    running_score = 0
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        batch_result = utils.compute_metrics(predictions=predictions.cpu(), references=batch["labels"].cpu())['f1']
        running_score += batch_result
    print("Evaluation Completed")
    print("F1: {}".format(running_score/len(test_dataloader)))