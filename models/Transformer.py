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
from Transformer.EncodingLayer import EncodingLayer
from Transformer.DecodingLayer import DecodingLayer


# Transformer model class:
class Model(nn.Module):
    def __init__(self, args, embeddings):
        super(Model, self).__init__()
        # Define embeddings (word and positional):
        self.embedding_dims = 300
        embeddings.append(torch.rand((self.embedding_dims,)))
        embeddings.insert(0, torch.zeros((self.embedding_dims,)))
        embedding_table = torch.stack(embeddings)
        self.embedding_length = len(embedding_table)
        self.embedding_word = torch.nn.Embedding.from_pretrained(
            embeddings=embedding_table,
            freeze=False,
            padding_idx=0,
        )
        self.embedding_positional = torch.nn.Embedding(
            num_embeddings=self.embedding_length,
            embedding_dim=300,
        )

        # Define parameters for model structure:
        self.s_a_unit_count = args.s_a_unit_count
        self.dimensions = args.dimensions
        self.embedding_dims = 300  # Using GloVe 300 dimensions, so can leave hard-coded
        self.ff_inner_dim = 1200   # In original paper inner dims = 4 * embedding dims
        self.layer_count = args.layer_count

        # Define Encoder block, using ModuleList:
        self.encoder = torch.nn.ModuleList(
            [
                EncodingLayer(
                    s_a_unit_count=self.s_a_unit_count,
                    dimensions=self.dimensions,
                    embedding_dims=self.embedding_dims,
                    ff_inner_dim=self.ff_inner_dim,
                ) for i in range(self.layer_count)
            ]
        )

        self.decoder = torch.nn.ModuleList(
            [
                DecodingLayer(
                    s_a_unit_count=self.s_a_unit_count,
                    dimensions=self.dimensions,
                    embedding_dims=self.embedding_dims,
                    ff_inner_dim=self.ff_inner_dim,
                ) for i in range(self.layer_count)
            ]
        )

        # Define final fully-connected layer (returns OHE sized vectors) and Softmax classifier:
        self.final_fc_layer = torch.nn.Linear(in_features=self.embedding_dims, out_features=self.embedding_length)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, x, y):
        # Send through embeddings to prepare data:
        x_embedded = self.embedding_word.forward(x)
        y_embedded = self.embedding_word.forward(y)

        # Positional encoding for inputs/outputs:
        x_positional = self.embedding_positional(x)
        y_positional = self.embedding_positional(y)
        x_encoded = x_embedded + x_positional
        y_encoded = y_embedded + y_positional

        # Go through encoder stack and get the result:
        for encoding_layer in self.encoder:
            x_encoded = encoding_layer.forward(x_encoded)

        # Now go through decoder stack and get result:
        # Put y_encoded in decoder_result, so can use same variable for recursive passing:
        decoder_result = y_encoded
        for decoding_layer in self.decoder:
            decoder_result = decoding_layer.forward(x_encoded, decoder_result)

        # Go through final layers (small eps to avoid NaN):
        out = self.final_fc_layer.forward(decoder_result)
        out = self.softmax.forward(out + 1e-16)
        return out
