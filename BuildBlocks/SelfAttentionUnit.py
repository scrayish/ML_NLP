"""

Self attention unit:
In features=embedding size, out features=dimensions parameter
Currently embedding size = 300 because possibly using GloVe embeddings (most likely)

"""


from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np


# Self attention class which can be imported and used:
class SelfAttentionUnit(nn.Module):
    def __init__(self, dimensions, masked: bool):
        super(SelfAttentionUnit, self).__init__()
        # Dimensions for scaling Q and K multiplication
        self.dimensions = dimensions
        # Introducing masking bool check for masked multi-head:
        self.masked = masked
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q_matrix, k_matrix, v_matrix, return_matrix=False):

        # Matrix multiplication between Q and K^T:
        q_k_matrix = torch.matmul(q_matrix, k_matrix.transpose_(2, 1))

        # Scaling result with square root value of inner dimensions:
        q_k_scaled = q_k_matrix / np.sqrt(self.dimensions)

        # Masking outputs (Optional)
        if self.masked:
            # Setting illegal (next-in-sequence) values to -Inf:
            mask = torch.tril(torch.ones(q_k_scaled.size(-1),
                                         q_k_scaled.size(-1))).view(1, q_k_scaled.size(-1), q_k_scaled.size(-1))
            q_k_scaled = q_k_scaled.masked_fill(mask == 0, value=float('-inf'))

        # Using softmax (Don't know about eps usage, maybe don't need to):
        q_k_softmaxed = self.softmax(q_k_scaled + 1e-16)

        matrix = None
        if return_matrix:
            matrix = q_k_softmaxed.detach()

        # Final matmul operation:
        out = torch.matmul(q_k_softmaxed, v_matrix)
        return out, matrix
