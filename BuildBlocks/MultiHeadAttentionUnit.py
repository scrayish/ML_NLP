"""

Multi-head attention unit for encoder/decoder:


"""


from __future__ import print_function
import torch
import torch.nn as nn
from BuildBlocks.SelfAttentionUnit import SelfAttentionUnit


# Multi-Head attention class:
class MultiHeadAttentionUnit(nn.Module):
    def __init__(self, s_a_unit_count, dimensions, embedding_dims, need_mask: bool):
        super(MultiHeadAttentionUnit, self).__init__()

        # All needed size parameters:
        self.s_a_unit_count = s_a_unit_count
        self.dimensions = dimensions
        self.need_mask = need_mask
        self.embedding_dims = embedding_dims

        # FFNs for splitting input to query, key, value:
        # Create ModuleList containing N linear layers (N = self attention unit count):
        self.value_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.embedding_dims, self.dimensions),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dimensions, self.dimensions),
            ) for i in range(self.s_a_unit_count)
        ])
        self.key_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.embedding_dims, self.dimensions),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dimensions, self.dimensions),
            ) for i in range(self.s_a_unit_count)
        ])
        self.query_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.embedding_dims, self.dimensions),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dimensions, self.dimensions),
            ) for i in range(self.s_a_unit_count)
        ])

        self.final_output_layer = torch.nn.Linear(in_features=self.s_a_unit_count * self.dimensions,
                                                  out_features=self.s_a_unit_count * self.dimensions)

        # Self-attention units in a ModuleList (So can create dynamic models):
        self.self_attention_units = torch.nn.ModuleList(
            [SelfAttentionUnit(dimensions=self.dimensions,
                               masked=self.need_mask) for i in range(self.s_a_unit_count)]
        )

    def forward(self, x, return_matrix=False):

        # List for all self attention unit outputs:
        all_unit_outputs = []
        matrix = None
        # Loop through all self attention units in MHA unit:
        for i in range(self.s_a_unit_count):

            # Get V, K, Q for individual self attention unit:
            v_unit = self.value_layers[i].forward(x)
            k_unit = self.key_layers[i].forward(x)
            q_unit = self.query_layers[i].forward(x)

            # Pass individual matrices to the unit itself and get result:
            unit_output, matrix = self.self_attention_units[i].forward(
                q_matrix=q_unit,
                k_matrix=k_unit,
                v_matrix=v_unit,
                return_matrix=return_matrix,
            )

            # Accumulate all outputs:
            all_unit_outputs.append(unit_output)

        # Concatenate all outputs into one and final linear layer
        all_unit_outputs_concat = torch.cat(all_unit_outputs, dim=2)
        out = self.final_output_layer.forward(all_unit_outputs_concat)
        return out, matrix
