"""

Transformer model based on paper "Attention is all you need"
Scuffed home-brew implementation

17.07.2020. ^^^ Have you no confidence in yourself boy? You should have plenty mate.
I've seen everything you've done and achieved despite opposition and conditions...
You must carry on. It's the only way.

-S


"""


from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from BuildBlocks.EncodingLayer import EncodingLayer
from BuildBlocks.DecodingLayer import DecodingLayer
from BuildBlocks.SinCosEmbedding import PositionalEncoding

# BuildBlocks model class:
class Model(nn.Module):
    def __init__(self, args, end_token, sine_position=False):
        super(Model, self).__init__()

        # Parameters for building Transformer:
        self.s_a_unit_count = args.s_a_unit_count
        self.dimensions = args.dimensions
        self.ff_inner_dim = self.dimensions * 4
        self.layer_count = args.layer_count

        # Embedding definitions:
        # Word embedding:
        self.embedding_word = torch.nn.Embedding(
            num_embeddings=end_token + 1,
            embedding_dim=self.dimensions,
            padding_idx=0,
        )

        # Positional embedding - can be either sin/cos or table (same as word):
        # Positional embedding is based on how long is longest sequence, need to know this beforehand;
        # Could leave headroom as is done here, as longest sequence is 93 words:
        if sine_position:
            self.embedding_positional = PositionalEncoding(
                num_embeddings=100,
                embedding_dim=self.dimensions,
            )
        else:
            self.embedding_positional = torch.nn.Embedding(
                num_embeddings=100,
                embedding_dim=self.dimensions,
                padding_idx=0,
            )

            # Disable gradient, in testing enabled gradient didn't really work:
            self.embedding_positional.requires_grad_(False)

        # Define Encoder:
        self.encoder = torch.nn.ModuleList([
            EncodingLayer(
                s_a_unit_count=self.s_a_unit_count,
                dimensions=int(self.dimensions / self.s_a_unit_count),
                embedding_dims=self.dimensions,
                ff_inner_dim=self.ff_inner_dim,
                need_mask=False,
            ) for i in range(self.layer_count)
        ])

        # Define Decoder:
        self.decoder = torch.nn.ModuleList([
            DecodingLayer(
                s_a_unit_count=self.s_a_unit_count,
                dimensions=int(self.dimensions / self.s_a_unit_count),
                embedding_dims=self.dimensions,
                ff_inner_dim=self.ff_inner_dim,
            ) for i in range(self.layer_count)
        ])

        self.final_linear = torch.nn.Linear(self.dimensions, end_token + 1)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, x, y, return_matrix):

        # Transform input to embeddings:
        index_matrix_x = np.asarray([
            np.arange(1, x.size(1) + 1) for i in range(x.size(0))
        ])
        index_matrix_y = index_matrix_x + 1
        remove_padded_grid = (x != 0).int().numpy()
        index_matrix_x = torch.LongTensor(
            index_matrix_x * remove_padded_grid
        ).to(x.device)
        x_embedded = self.embedding_word.forward(x)
        x_encoded = x_embedded * self.embedding_positional.forward(index_matrix_x)

        # Pass values through embeddings, pass positions through embeddings and add together:
        # Configure y to all ones for freedom of movement for rollout transformer:
        if y is None:
            y_encoded = torch.ones_like(x_encoded)
        else:
            index_matrix_y = torch.LongTensor(
                index_matrix_y * remove_padded_grid
            ).to(x.device)
            y_embedded = self.embedding_word.forward(y)
            y_encoded = y_embedded * self.embedding_positional.forward(index_matrix_y)

        # Pass x values through encoder part:
        matrix = None
        for layer in self.encoder:
            x_encoded, matrix = layer.forward(x_encoded, return_matrix)

        # Pass values into decoder part:
        y_out = y_encoded
        for layer in self.decoder:
            y_out, matrix = layer.forward(x_encoded, y_out)

        # Final linear layer and softmax over it:
        out = self.final_linear(y_out)
        out = self.softmax(out)
        return out, matrix
