import os


seed = 42

# Directory parameters
data_dir = os.path.expanduser('data/patentsview/example')
root_dir = os.path.expanduser("experiments")

# Tokenizer parameters.
max_length = 4096 #16384
load_saved_tokens = False
save_tokens = True

# Model parameters.
labels_list = ["A","B","C","D","E","F","G","H"]
num_labels = 8
small_scale = True

# Training parameters.
batch_size = 2#4
lr= 5e-5
weight_decay=1e-2
num_warmup_steps=0
num_epochs = 1

