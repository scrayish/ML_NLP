"""

All models for using this program:


"""
from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


# Model No.1: One directional LSTM RNN
class Model1(nn.Module):
    def __init__(self, batch_size, vocab_len, hidden_size, emb_dims):
        super(Model1, self).__init__()

        self.hidden_size = hidden_size
        self.word_count = vocab_len
        self.embedding_dims = emb_dims
        self.batch_size = batch_size

        # Define embedding:
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.word_count,
            embedding_dim=self.embedding_dims,
            padding_idx=0
        )

        self.lstm = torch.nn.LSTM(
            batch_first=True,
            input_size=self.embedding_dims,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True)

        self.fc = torch.nn.Linear(in_features=self.hidden_size, out_features=self.word_count)
        self.out_tanh = torch.nn.Tanh()

        # Weight and bias initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            if 'bias' in name:
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
        out = self.fc.forward(out.data)
        #out = self.stable_softmax(out)
        out = torch.softmax(out + 1e-16, dim=1)

        out = PackedSequence(
            out,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices,
            unsorted_indices=x.unsorted_indices
        )

        return out, h


# Model No.2: One directional LSTM RNN with extra embedding pass through layer
class Model2(nn.Module):
    def __init__(self, batch_size, vocab_len, hidden_size, emb_dims):
        super(Model2, self).__init__()

        self.hidden_size = hidden_size
        self.word_count = vocab_len
        self.embedding_dims = emb_dims
        self.batch_size = batch_size

        # Define embedding:
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.word_count,
            embedding_dim=self.embedding_dims,
            padding_idx=0
        )

        self.lstm = torch.nn.LSTM(
            batch_first=True,
            input_size=self.embedding_dims,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True)

        self.fc = torch.nn.Linear(in_features=self.hidden_size, out_features=self.embedding_dims)
        self.out_tanh = torch.nn.Tanh()

        # Weight and bias initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            if 'bias' in name:
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
        out = self.fc.forward(out.data)

        # Transposed embedding weights to give more precision
        out = torch.matmul(out, self.embedding.weight.t())

        out = torch.softmax(out + 1e-16, dim=1)

        out = PackedSequence(
            out,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices,
            unsorted_indices=x.unsorted_indices
        )

        return out, h


# Model No.3: Bi-directional LSTM RNN
class Model3(nn.Module):
    def __init__(self, batch_size, vocab_len, hidden_size, emb_dims):
        super(Model3, self).__init__()

        self.hidden_size = hidden_size
        self.word_count = vocab_len
        self.embedding_dims = emb_dims
        self.batch_size = batch_size

        # Define embedding:
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.word_count,
            embedding_dim=self.embedding_dims,
            padding_idx=0
        )

        self.lstm = torch.nn.LSTM(
            batch_first=True,
            input_size=self.embedding_dims,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            bidirectional=True
        )

        self.fc = torch.nn.Linear(in_features=self.hidden_size * 2, out_features=self.embedding_dims)
        self.out_tanh = torch.nn.Tanh()

        # Weight and bias initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            if 'bias' in name:
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
        out = self.fc.forward(out.data)

        # Transposed embedding weights to give more precision
        out = torch.matmul(out, self.embedding.weight.t())

        out = torch.softmax(out + 1e-16, dim=1)

        out = PackedSequence(
            out,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices,
            unsorted_indices=x.unsorted_indices
        )

        return out, h

