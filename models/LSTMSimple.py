"""

One directional RNN using one LSTM cell for computation


"""

from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.parameter import Parameter


class Model(nn.Module):
    def __init__(self, args, word_count, embeddings):
        super(Model, self).__init__()

        self.hidden_size = args.hidden_size
        # Need +1 for padding embedding IMO
        self.word_count = word_count
        # Hard-coded because of pre-trained embeddings
        self.embedding_dims = 300
        self.batch_size = args.batch_size
        embeddings.append(torch.rand((self.embedding_dims, )))
        embedding_table = torch.stack(embeddings)

        # Define embedding:
        self.embedding = torch.nn.Embedding.from_pretrained(
            embeddings=embedding_table,
            freeze=False,
        )

        self.lstm = torch.nn.LSTM(
            batch_first=True,
            input_size=self.embedding_dims,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
        )

        self.fc = torch.nn.Linear(in_features=self.hidden_size, out_features=self.embedding_dims)
        self.fc1 = torch.nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.fc2 = torch.nn.Linear(in_features=self.hidden_size, out_features=self.embedding_dims)
        self.fc3 = torch.nn.Linear(in_features=self.embedding_dims, out_features=self.embedding_dims)

        # Weight and bias initialization

        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'embedding' in name:
                    continue
                nn.init.xavier_normal_(param)
            if 'lstm.bias' in name:
                n = param.size(0)
                start, end = n//4, n//2
                param.data[start:end].fill_(1.0)

    def forward(self, x: PackedSequence, h):

        # (batch, seq*features)

        x = PackedSequence(
            self.embedding.forward(x.data),
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices,
            unsorted_indices=x.unsorted_indices
        )

        # Padding sequence
        #x = torch.nn.utils.rnn.pad_packed_sequence(x)

        # (Batch, seq, features)
        out, h = self.lstm.forward(x, h)
        # Stiches multiple out segments together in one memory space
        # No use when PackedSequence
        # out = out.contiguous()
        # Same functionality as np.reshape - packed sequence has no .view
        # out = out.view(-1, self.h_s)
        out = self.fc1.forward(out.data)
        out = self.fc2.forward(out)
        out = torch.matmul(out, self.embedding.weight.t())
        #out = self.stable_softmax(out)
        out = torch.softmax(out + 1e-16, dim=1)

        out = PackedSequence(
            out,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices,
            unsorted_indices=x.unsorted_indices
        )

        return out, h
