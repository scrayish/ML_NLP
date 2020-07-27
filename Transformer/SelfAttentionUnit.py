"""

Self attention unit:
In features=embedding size, out features=dimensions parameter
Currently embedding size = 300 because possibly using GloVe embeddings (most likely)

"""


from __future__ import print_function
import torch
import torch.nn as nn


# Self attention class which can be imported and used:
class SelfAttentionUnit(nn.Module):
    def __init__(self, dimensions, masked: bool):
        super(SelfAttentionUnit, self).__init__()
        # Dimensions for scaling Q and K multiplication
        self.dimensions = dimensions
        # Introducing masking bool check for masked multi-head:
        self.masked = masked
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q_matrix, k_matrix, v_matrix):

        # Matrix multiplication between Q and K^T:
        q_k_matrix = torch.matmul(q_matrix, k_matrix.transpose_(2, 1))

        # Scaling result:
        q_k_scaled = q_k_matrix / self.dimensions
        # TODO: Return scaled matrix out for inspection graphically

        # Masking outputs (Optional)
        if self.masked:
            # Setting illegal (next-in-sequence) values to -Inf:
            q_k_scaled = torch.tril(q_k_scaled, diagonal=0)
            q_k_scaled.masked_fill_(q_k_scaled == 0, value=float('-inf'))

        # Using softmax (Don't know about eps usage, maybe don't need to):
        q_k_softmaxed = self.softmax(q_k_scaled + 1e-16)

        # Final matmul operation:
        out = torch.matmul(q_k_softmaxed, v_matrix)
        return out
