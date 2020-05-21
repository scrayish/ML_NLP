"""

Class for holding utility functions

"""
from __future__ import print_function
import torch
import numpy as np
from sklearn.metrics import f1_score


class Utility(object):
    def __init__(self):
        pass

    @staticmethod
    def f1score(y, y_prim):
        y_true = np.zeros((len(y),))
        y_pred = np.zeros((len(y_prim),))

        for i in range(len(y_prim)):
            y_true[i] = y[i].argmax()
            y_pred[i] = y_prim[i].argmax()

        return np.mean(f1_score(y_true, y_pred, average='macro'))

        # Data labeling:

    @staticmethod
    def words_to_label(quotes, vocabulary):
        quotes_labeled = []
        for quote in quotes:
            words = quote.split()
            s_q_labeled = []

            for word in words:
                s_q_labeled.append(vocabulary[word])

            s_q_labeled = torch.LongTensor(np.asarray(s_q_labeled))
            quotes_labeled.append(s_q_labeled)

        return quotes_labeled

    # Function that can work with dynamic length sequences: (HP required above this function)
    @staticmethod
    def collate_fn(batch_input):
        x_input, y_input, x_len = zip(*batch_input)

        x_len_max = int(np.max(x_len))

        # sort batch so that max x_len first (descending)
        input_indexes_sorted = list(reversed(np.argsort(x_len).tolist()))

        # x = torch.zeros((batch_size, x_len_max), dtype=torch.long)
        # y = torch.zeros_like(x)

        x = []
        y = []

        x_len_out = torch.LongTensor(x_len)

        for i in range(len(batch_input)):
            i_sorted = input_indexes_sorted[i]
            x_len_out[i] = x_len[i_sorted]
            x.append(x_input[i_sorted])
            y.append(y_input[i_sorted])

            # x[i, 0:x_len_out[i]] = x_input[i_sorted]
            # y[i, 0:x_len_out[i]] = y_input[i_sorted]

        return (x, y, x_len_out)

    @staticmethod
    def data_weight_coefficients(word_count, total_word_count):
        word_coefficients = []

        for word in word_count:
            word_coefficients.append(word_count[word])

        word_coefficients = np.array(word_coefficients)
        word_coefficients = 1 - word_coefficients / total_word_count
        word_coefficients = np.append(word_coefficients, 1 / int(total_word_count / 2))
        word_coefficients = torch.FloatTensor(word_coefficients)
        return word_coefficients

    @staticmethod
    def list_split():
        pass