import config
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(predictions, references):

    precision, recall, f1, _ = precision_recall_fscore_support(references, predictions, average='micro')
    acc = accuracy_score(references, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def attention_mapper(device):
    global_attention_mask = torch.zeros([config.batch_size, config.max_length], dtype=torch.long, device=device)
    return global_attention_mask