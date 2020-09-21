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


# BuildBlocks model class:
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
        # Currently hard-coding, no need to get big values since not many long sequences and not too much to context:
        self.embedding_positional = torch.nn.Embedding(
            num_embeddings=100,
            embedding_dim=300,
            padding_idx=0,
        )

        # Define parameters for model structure:
        self.s_a_unit_count = args.s_a_unit_count
        self.dimensions = args.dimensions
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
                    need_mask=True,
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

    def forward(self, x, return_matrix):
        # Send through embeddings to prepare data:
        # Get indices for values in sentences to apply positional encoding:
        sentence_nums_x = []
        sentence_nums_y = []
        for sequence in x:

            # Build sequence using the length of sequence:
            sequence_positional_indices_x = []
            sequence_positional_indices_y = []
            for idx, token in enumerate(sequence, 1):
                if token == 0:
                    sequence_positional_indices_x.append(0)
                    sequence_positional_indices_y.append(0)
                else:
                    sequence_positional_indices_x.append(idx)
                    sequence_positional_indices_y.append(idx + 1)

            sequence_positional_indices_x = np.asarray(sequence_positional_indices_x, dtype=np.int32)
            sequence_positional_indices_y = np.asarray(sequence_positional_indices_y, dtype=np.int32)
            sentence_nums_x.append(sequence_positional_indices_x)
            sentence_nums_y.append(sequence_positional_indices_y)

        # Turn all of them into tensors for passing through embedding for position:
        sentence_nums_x = torch.LongTensor(sentence_nums_x)
        sentence_nums_y = torch.LongTensor(sentence_nums_y)

        # Embed x values:
        x_embedded = self.embedding_word.forward(x)

        # Positional encoding for inputs/outputs:
        x_positional = self.embedding_positional(sentence_nums_x)
        y_positional = self.embedding_positional(sentence_nums_y)
        x_encoded = x_embedded + x_positional

        # Go through encoder stack and get the result:
        for encoding_layer in self.encoder:
            x_encoded, matrix = encoding_layer.forward(x_encoded, return_matrix)

        # Now go through decoder stack and get result:
        # Apply positional encoding to encoder output for first "output" input in decoder:
        # Put y_encoded in decoder_result, so can use same variable for recursive passing:
        decoder_result = x_encoded + y_positional
        for decoding_layer in self.decoder:
            decoder_result = decoding_layer.forward(x_encoded, decoder_result)

        # Go through final layers (small eps to avoid NaN):
        out = self.final_fc_layer.forward(decoder_result)
        out = self.softmax.forward(out + 1e-16)
        return out, matrix
