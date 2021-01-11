"""

Full encoding layer, utilizing multi-head attention with FFNN:
This layer contains matrices for getting Q, K, V vectors (Query, Key, Value)


"""


from __future__ import print_function
import torch
import torch.nn as nn
from BuildBlocks.MultiHeadAttentionUnit import MultiHeadAttentionUnit


# Encoding layer class:
class EncodingLayer(nn.Module):
    def __init__(self, s_a_unit_count, dimensions, embedding_dims, ff_inner_dim, need_mask: bool):
        super(EncodingLayer, self).__init__()
        # All size parameters:
        self.s_a_unit_count = s_a_unit_count
        self.dimensions = dimensions
        self.embedding_dims = embedding_dims
        self.ff_inner_dim = ff_inner_dim
        self.masked = need_mask

        # Define layer normalization (LayerNorm) https://arxiv.org/pdf/1607.06450.pdf:
        self.layer_norm_1 = torch.nn.LayerNorm(normalized_shape=self.embedding_dims)
        self.layer_norm_2 = torch.nn.LayerNorm(normalized_shape=self.embedding_dims)

        # Multi-head attention unit:
        self.multi_head_attention_unit = MultiHeadAttentionUnit(
            s_a_unit_count=self.s_a_unit_count,
            dimensions=self.dimensions,
            embedding_dims=self.embedding_dims,
            need_mask=self.masked,
        )

        # Feed forward neural network layers:
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.embedding_dims, out_features=self.ff_inner_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self.ff_inner_dim, out_features=self.embedding_dims),
        )

    def forward(self, x, return_matrix):
        # Send data matrix into multi-head attention (Permute for max sequence length to be 1st dimension):
        mha_result, matrix = self.multi_head_attention_unit.forward(
            x=x,
            return_matrix=return_matrix,
        )

        # Sum Multi-head attention result with input and normalize (Permute back previous change):
        ff_input = self.layer_norm_1(mha_result + x)

        # Feed-Forward Neural network:
        out = self.feed_forward.forward(ff_input)

        # Sum result, normalize and pass out of the layer:
        out = self.layer_norm_2(out + ff_input)
        return out, matrix
