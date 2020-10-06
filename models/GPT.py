"""

NLP model based on GPT architecture
Stealing encoding layer from transformer as it is same principle, as to not duplicate code

"""


from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from BuildBlocks.EncodingLayer import EncodingLayer


# GPT Model class:
class Model(nn.Module):
    def __init__(self, args, end_token):
        super(Model, self).__init__()

        # Define embeddings (embedding dims = 300 to be comparable with GloVe embeddings):
        self.embedding_word = torch.nn.Embedding(
            num_embeddings=end_token + 1,
            embedding_dim=300,
            padding_idx=0,
        )
        self.embedding_positional = torch.nn.Embedding(
            num_embeddings=100,
            embedding_dim=300,
            padding_idx=0,
        )

        # Freeze positional embedding:
        self.embedding_positional.weight.requires_grad = False

        # Extra variables:
        self.s_a_unit_count = args.s_a_unit_count
        self.dimensions = args.dimensions
        # In original paper inner dims = 4 * embedding_dims:
        self.ff_inner_dim = self.s_a_unit_count * self.dimensions * 4
        self.layer_count = args.layer_count

        # Define encoder:
        self.encoder = torch.nn.ModuleList([
            EncodingLayer(
                s_a_unit_count=self.s_a_unit_count,
                dimensions=self.dimensions,
                embedding_dims=300,
                ff_inner_dim=self.ff_inner_dim,
                need_mask=True,
            ) for i in range(self.layer_count)
        ])

        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, x, return_matrix):
        # First get embeddings for input:
        # Send through embeddings to prepare data:
        # Get indices for values in sentences to apply positional encoding:
        sentence_nums_x = []
        for sequence in x:

            # Build sequence using the length of sequence:
            sequence_positional_indices_x = []
            for idx, token in enumerate(sequence, 1):
                if token == 0:
                    sequence_positional_indices_x.append(0)
                else:
                    sequence_positional_indices_x.append(idx)

            sequence_positional_indices_x = np.asarray(sequence_positional_indices_x, dtype=np.int32)
            sentence_nums_x.append(sequence_positional_indices_x)

        # Turn all of them into tensors for passing through embedding for position:
        sentence_nums_x = torch.LongTensor(sentence_nums_x)

        # Get embeddings and set up working principles:
        x_embedded = self.embedding_word.forward(x)
        x_positional = self.embedding_positional.forward(sentence_nums_x.to(x.device))
        x_encoded = x_embedded + x_positional

        matrix = None
        for i in range(len(self.encoder)):
            x_encoded, matrix = self.encoder[i].forward(x_encoded, return_matrix)

        # Multiplication with transposed embedding:
        out = torch.matmul(x_encoded, self.embedding_word.weight.t())
        out = self.softmax.forward(out)
        return out, matrix
