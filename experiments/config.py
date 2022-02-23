from transformers import TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

num_labels = 8
batch_size = 2

CustomTrainingArguments = TrainingArguments(

    logging_dir='/logs',
    logging_strategy = 'steps',
    logging_steps = 1,

    evaluation_strategy = 'epoch',
    save_strategy='epoch',
    # save_steps = '1' if save_strategy is steps

    num_train_epochs = 1,
    learning_rate = 5e-5,
    weight_decay=0.01,
    # warmup_ratio = 0.1

    per_device_train_batch_size = 2,   
    per_device_eval_batch_size= 2,

    # gradient_accumulation_steps = 1, 
    
    dataloader_num_workers = 0,

    disable_tqdm = False, 
    load_best_model_at_end=True,
    fp16 = False,

    seed = 42,
    run_name = 'lonngformer_finetuning',
    output_dir = '/results',
    )


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }