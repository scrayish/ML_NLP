"""

Multi-head attention unit for encoder/decoder:


"""


from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from BuildBlocks.SelfAttentionUnit import SelfAttentionUnit


# Multi-Head attention class:
class MultiHeadAttentionUnit(nn.Module):
    def __init__(self, unit_count, dimensions, embedding_dims, need_mask: bool):
        super(MultiHeadAttentionUnit, self).__init__()

        # All needed size parameters:
        self.self_attention_units = unit_count
        self.dimensions = dimensions
        self.need_mask = need_mask
        self.embedding_dims = embedding_dims

        # Linear layers for multi-head:
        # Create layers N times the size (N = heads), then split in N equal parts:
        self.value_layer = torch.nn.Linear(in_features=self.dimensions,
                                           out_features=self.self_attention_units * self.dimensions)
        self.key_layer = torch.nn.Linear(in_features=self.dimensions,
                                         out_features=self.self_attention_units * self.dimensions)
        self.query_layer = torch.nn.Linear(in_features=self.dimensions,
                                           out_features=self.self_attention_units * self.dimensions)
        self.final_output_layer = torch.nn.Linear(in_features=self.self_attention_units * self.dimensions,
                                                  out_features=self.embedding_dims)

        # Self-attention units in a ModuleList (So can create dynamic models):
        self.SelfAttentionUnits = torch.nn.ModuleList(
            [SelfAttentionUnit(dimensions=self.dimensions,
                               masked=self.need_mask) for i in range(self.self_attention_units)]
        )

    def forward(self, q_matrix, k_matrix, v_matrix, return_matrix=False):

        # Let all matrices through linears, accumulate all values:
        all_q_matrix_values = self.query_layer(q_matrix)
        all_k_matrix_values = self.key_layer(k_matrix)
        all_v_matrix_values = self.value_layer(v_matrix)

        # List to accumulate all outputs:
        all_unit_outputs = []

        # Indices for iterating over all values from respective matrices:
        start_index = 0
        end_index = self.dimensions

        # Iterate through all Self-attention units and receive their outputs:
        matrix = None
        for unit in self.SelfAttentionUnits:
            unit_output, matrix = unit.forward(
                all_q_matrix_values[:, :, start_index:end_index],
                all_k_matrix_values[:, :, start_index:end_index],
                all_v_matrix_values[:, :, start_index:end_index],
                return_matrix,
            )
            all_unit_outputs.append(unit_output)
            start_index += self.dimensions
            end_index += self.dimensions

        # Concatenate all outputs into one and final linear layer
        all_unit_outputs_concat = torch.cat(all_unit_outputs, dim=2)
        out = self.final_output_layer.forward(all_unit_outputs_concat)
        return out, matrix
