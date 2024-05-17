import torch


batch_size = 64
block_size = 528
max_iters = 10_000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
n_head = 8
n_layer = 6
dropout = 0.3