import os
import config
import pandas as pd

from tqdm.auto import tqdm

from accelerate import Accelerator

import torch
from torch.utils.data import DataLoader

from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, LongformerConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler

from datasets import Features, Value, ClassLabel, load_dataset


def batch_tokenizer(batch):
    return tokenizer(batch["text"], padding='max_length', truncation=True)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_dir = os.path.expanduser('data/patentsview/example')
small_scale = True


if __name__ == '__main__':

    torch.cuda.empty_cache()
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
        train_data = tokenized_data["train"].shuffle(seed=config.seed).select(range(32))
        test_data = tokenized_data["test"].shuffle(seed=config.seed).select(range(16))
    else:
        train_data = tokenized_data["train"]
        test_data = tokenized_data["test"]

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_data, shuffle=True, batch_size=config.batch_size)

    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
        num_labels = config.num_labels,
        gradient_checkpointing=True        
        )

    # config = LongformerConfig()

    optimizer = AdamW(model.parameters(), lr=config.lr)

    accelerator = Accelerator()
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

    model.train()
    for epoch in range(config.num_epochs):
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
    """metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()"""