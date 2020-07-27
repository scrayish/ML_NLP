"""

One directional RNN using two GRU cells for computation

Using GloVe pre-trained embeddings as per training configuration

"""

from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


class Model(nn.Module):
    def __init__(self, args, word_count, embeddings):
        super(Model, self).__init__()

        self.hidden_size = args.hidden_size
        self.word_count = word_count
        # Cieti noteikta vērtība, jo izmantoti iepriekš trenētas vērtības
        self.embedding_dims = 300
        embeddings.append(torch.rand((self.embedding_dims,)))
        embeddings.insert(0, torch.zeros(self.embedding_dims, ))
        embedding_table = torch.stack(embeddings)

        # Definē iegultās vērtības
        self.embedding = torch.nn.Embedding.from_pretrained(
            embeddings=embedding_table,
            freeze=False,
            padding_idx=0,
        )

        self.gru = torch.nn.GRU(
            batch_first=True,
            input_size=self.embedding_dims,
            hidden_size=self.hidden_size,
            num_layers=2,
            bias=True,
            dropout=0.5,
        )

        self.fc1 = torch.nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.fc2 = torch.nn.Linear(in_features=self.hidden_size, out_features=self.embedding_dims)

        # Svaru un nobīdes inicializācija
        for name, param in self.named_parameters():
            if 'weight' in name:
                # Atkārtoti neinicializē iegulto vērtību svarus!
                if 'embedding' in name:
                    continue
                nn.init.xavier_normal_(param)
            if 'lstm.bias' in name:
                n = param.size(0)
                start, end = n//4, n//2
                param.data[start:end].fill_(1.0)

    def forward(self, x: PackedSequence, h):

        x = PackedSequence(
            self.embedding.forward(x.data),
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices,
            unsorted_indices=x.unsorted_indices
        )

        out, h = self.gru.forward(x, h)
        out = self.fc1.forward(out.data)
        out = self.fc2.forward(out)
        out = torch.matmul(out, self.embedding.weight.t())
        out = torch.softmax(out + 1e-16, dim=1)

        out = PackedSequence(
            out,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices,
            unsorted_indices=x.unsorted_indices
        )

        return out, h
