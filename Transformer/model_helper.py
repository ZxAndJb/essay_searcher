import torch
import torch.nn as nn


def generate_embeddings(vocab_size, hidden_dim, padding_idx):
    # the vector corresponding to the padding_idx will not involve in the computation of gradient and gradient update
    m = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=hidden_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m