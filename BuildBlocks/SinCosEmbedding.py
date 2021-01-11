"""

Sin/Cos Positional embedding for GPT/Transformer models:


"""

import torch
import torch.nn as nn
import numpy as np


# Sin/Cos positional encoding:
class PositionalEncoding(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super(PositionalEncoding, self).__init__()

        self.pe = torch.zeros(num_embeddings, embedding_dim)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, idxes):
        return self.pe[idxes, :]
