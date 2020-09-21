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
        self.s_a_units = s_a_unit_count
        self.dimensions = dimensions
        self.embedding_dims = embedding_dims
        self.ff_inner_dimension = ff_inner_dim
        self.masked = need_mask

        # Q, K, V matrices as linear layers:
        self.embedding_to_query = torch.nn.Linear(in_features=self.embedding_dims, out_features=self.dimensions)
        self.embedding_to_key = torch.nn.Linear(in_features=self.embedding_dims, out_features=self.dimensions)
        self.embedding_to_value = torch.nn.Linear(in_features=self.embedding_dims, out_features=self.dimensions)

        # Define layer normalization (LayerNorm) https://arxiv.org/pdf/1607.06450.pdf:
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=self.embedding_dims)

        # Multi-head attention unit:
        self.multi_head_attention_unit = MultiHeadAttentionUnit(
            unit_count=self.s_a_units,
            dimensions=self.dimensions,
            embedding_dims=self.embedding_dims,
            need_mask=self.masked,
        )

        # Feed forward neural network layers:
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.embedding_dims, out_features=self.ff_inner_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self.ff_inner_dimension, out_features=self.embedding_dims),
        )

    def forward(self, x, return_matrix):

        # Turn embeddings into queries/keys/values:
        q_matrix = self.embedding_to_query.forward(x)
        k_matrix = self.embedding_to_key.forward(x)
        v_matrix = self.embedding_to_value.forward(x)

        # Send matrices into multi-head attention:
        mha_result, matrix = self.multi_head_attention_unit.forward(
            q_matrix=q_matrix,
            k_matrix=k_matrix,
            v_matrix=v_matrix,
            return_matrix=return_matrix,
        )

        # Sum Multi-head attention result with input and normalize:
        ff_input = self.layer_norm(mha_result + x)

        # Feed-Forward Neural network:
        out = self.feed_forward.forward(ff_input)

        # Sum result, normalize and pass out of the layer:
        out = self.layer_norm(out + ff_input)
        return out, matrix
