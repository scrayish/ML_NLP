"""

Decoding layer for Transformer model:
Currently only base for building the model

"""


from __future__ import print_function
import torch
import torch.nn as nn
from BuildBlocks.MultiHeadAttentionUnit import MultiHeadAttentionUnit


# Decoding layer class:
class DecodingLayer(nn.Module):
    def __init__(self, s_a_unit_count, dimensions, embedding_dims, ff_inner_dim):
        super(DecodingLayer, self).__init__()
        # All size parameters:
        self.s_a_units = s_a_unit_count
        self.dimensions = dimensions
        self.embedding_dims = embedding_dims
        self.ff_inner_dimension = ff_inner_dim

        self.embedding_to_query_2 = torch.nn.Linear(in_features=self.embedding_dims, out_features=self.dimensions)
        self.embedding_to_key_2 = torch.nn.Linear(in_features=self.embedding_dims, out_features=self.dimensions)
        self.embedding_to_value_2 = torch.nn.Linear(in_features=self.embedding_dims, out_features=self.dimensions)

        # Define layer normalization (LayerNorm) https://arxiv.org/pdf/1607.06450.pdf:
        self.layer_norm_1 = torch.nn.LayerNorm(normalized_shape=self.embedding_dims)
        self.layer_norm_2 = torch.nn.LayerNorm(normalized_shape=self.embedding_dims)

        # Multi-head attention unit:
        self.multi_head_attention_unit = MultiHeadAttentionUnit(
            unit_count=self.s_a_units,
            dimensions=self.dimensions,
            embedding_dims=self.embedding_dims,
            need_mask=False,
        )

        # Masked multi-head attention unit:
        self.masked_multi_head_attention_unit = MultiHeadAttentionUnit(
            unit_count=self.s_a_units,
            dimensions=self.dimensions,
            embedding_dims=self.embedding_dims,
            need_mask=True,
        )

        # Feed forward neural network layers:
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.embedding_dims, out_features=self.ff_inner_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self.ff_inner_dimension, out_features=self.embedding_dims),
        )

    def forward(self, encoder_x, y):

        multi_head_result, matrix = self.masked_multi_head_attention_unit.forward(
            x=y,
            return_matrix=False,
        )

        # Sum result with starting input and normalize:
        normalized_result = self.layer_norm_1.forward(y + multi_head_result)

        # Sum encoder result and normalized result to combine them for second MHA unit:
        multi_head_result, matrix = self.multi_head_attention_unit.forward(
            x=encoder_x + normalized_result,
            return_matrix=False,
        )

        # Sum result with previous unit output and normalize:
        normalized_result = self.layer_norm.forward(multi_head_result + normalized_result)

        # Go through Feed-Forward Neural Network, then add input, normalize, return result:
        out = self.feed_forward.forward(normalized_result)
        out = self.layer_norm.forward(out + normalized_result)
        return out
