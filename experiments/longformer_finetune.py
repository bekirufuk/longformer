import os
import config, utils
import pandas as pd

from tqdm.auto import tqdm

from accelerate import Accelerator

import torch
from torch.utils.data import DataLoader

from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, LongformerConfig, get_scheduler, logging

from datasets import Features, Value, ClassLabel, load_dataset, load_from_disk


def batch_tokenizer(batch):
    return tokenizer(batch["text"],
    padding='max_length',
    truncation=True
    )


if __name__ == '__main__':
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.empty_cache()
    accelerator = Accelerator(fp16=True)

    # Load the dataset from csv file with proper features.
    class_names = config.labels_list
    features = Features({'text': Value('string'), 'label': ClassLabel(names=class_names)})
    data_files = {"train":os.path.join(config.data_dir, 'train.csv'), "test":os.path.join(config.data_dir, 'test.csv')}
    dataset = load_dataset('csv', data_files=data_files, features=features)
    
    # Load the existing tokenized data if wanted. Create a new one otherwise.
    if config.load_saved_tokens:
        tokenized_data = load_from_disk(os.path.join(config.data_dir, "tokenized_data"))
    else:
        # Utilize the tokenizer and run it on the dataset with batching.
        tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length=config.max_length)
        tokenized_data = dataset.map(batch_tokenizer, batched=True, remove_columns=['text'])
        tokenized_data = tokenized_data.rename_column("label","labels")
        tokenized_data.set_format("torch")
    
    if config.save_tokens:
        tokenized_data.save_to_disk(os.path.join(config.data_dir, "tokenized_data"))

    #Trim the dataset to a small size for testing purposes.
    if config.small_scale:
        train_data = tokenized_data["train"].shuffle(seed=config.seed).select(range(16))
        test_data = tokenized_data["test"].shuffle(seed=config.seed).select(range(8))
    else:
        train_data = tokenized_data["train"]
        test_data = tokenized_data["test"]

    # Dataloaders for PyTorch implementation.
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=config.batch_size)

    # Utilize the model with custom config file specifiyng classification labels.
    longformer_config = LongformerConfig.from_json_file(os.path.join(config.root_dir, 'longformer_config.json'))
    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
        config=longformer_config
    )
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Load the data with accelerator for a better GPU performence. No need to send it into the device.
    train_dataloader, test_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, test_dataloader, model, optimizer
    )
    
    # Epoch times number of batches gives the total step count. Feed it into tqdm for a progress bar with this many steps.
    num_training_steps = config.num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    # Scheduler for dynamic learning rate.
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Construct a global_attention_mask to decide which tokens will have glabal attention.
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
    print("\n----------\n TRAINING FINISHED \n----------\n")

    model.eval()
    print("\n----------\n EVALUATION STARTED \n----------\n")
    running_score = 0
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        batch_result = utils.compute_metrics(predictions=predictions.cpu(), references=batch["labels"].cpu())['f1']
        running_score += batch_result
    print("\n----------\n EVALUATION FINISHED \n----------\n")

    print("mean of {} batches F1: {}".format(len(test_dataloader),running_score/len(test_dataloader)))