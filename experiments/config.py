from transformers import TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

seed = 42

labels_list = ["A","B","C","D","E","F","G","H"]
num_labels = 8

batch_size = 4
max_length = 4096 #16384

lr= 5e-5
num_warmup_steps=0
num_epochs = 1


def compute_metrics(predictions, references):

    precision, recall, f1, _ = precision_recall_fscore_support(references, predictions, average='micro')
    acc = accuracy_score(references, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }