"""

Class for holding utility functions

"""
from __future__ import print_function
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from collections import Counter


class Utility(object):

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
    def words_to_label(quotes, vocabulary, rollout=False):
        # If passing train/test data
        if not rollout:
            quotes_labeled = []
            for quote in quotes:
                words = quote.split()
                s_q_labeled = []

                for word in words:
                    s_q_labeled.append(vocabulary[word])

                s_q_labeled = torch.LongTensor(np.asarray(s_q_labeled))
                quotes_labeled.append(s_q_labeled)

            return quotes_labeled
        # If passing data for rollout
        else:
            # If passed a sequence of words
            if len(quotes.split(' ')) > 1:
                sequence_labeled = []
                for word in quotes.split():
                    sequence_labeled.append(vocabulary[word])

                sequence_labeled = torch.LongTensor(np.asarray(sequence_labeled))
                return sequence_labeled
            # If passed one word
            else:
                word = []
                word.append(vocabulary[quotes])
                word = torch.squeeze(torch.LongTensor(np.asarray(word)))
                return word

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
    def data_weights(word_count, total_word_count, quote_count):
        word_weights = []
        for word in word_count:
            word_weight = 1.0 - word_count[word] / total_word_count
            word_weights.append(word_weight)

        word_weight = 1.0 - quote_count / total_word_count
        word_weights.append(word_weight)
        np_word_weights = np.array(word_weights)
        #np_word_weights /= np.sum(np_word_weights)
        t_word_weights =torch.FloatTensor(np_word_weights)
        return t_word_weights

    @staticmethod
    def draw_histograms(all_quotes, vocabulary, word_count):
        matplotlib.use("TkAgg")
        # Count how many times each word shows up in quotes and draw histogram:
        quote_len_hist = []
        for quote in all_quotes:
            quote_len_hist.append(len(quote.split()))

        words = []
        w_count = []
        for i in vocabulary:
            words.append(i)
        for i in word_count:
            w_count.append(word_count[i])
        words = np.array(words)
        w_count = np.array(w_count)
        w_count_most = w_count[:100]
        w_count_least = w_count[-101:-1]
        words_most = words[:100]
        words_least = words[-101:-1]
        quote_len_hist = Counter(quote_len_hist)
        quote_len_hist = sorted(quote_len_hist.items())

        # Splits off words from their count
        # TOP 100 most frequent words
        indices = np.arange(len(words_most))
        plt.figure(figsize=(16.0, 9.0))
        plt.title("100 Visbiežāk sastopamie vārdi datos")
        plt.bar(indices, w_count_most)
        plt.xticks(ticks=indices, labels=words_most, rotation=90)
        plt.xlabel("Vārdi")
        plt.ylabel("Vārdu skaits")

        # TOP 100 least frequent words
        indices = np.arange(len(words_least))
        plt.figure(figsize=(16.0, 9.0))
        plt.title("100 Visretāk sastopamie vārdi datos")
        plt.bar(indices, w_count_least)
        plt.xticks(ticks=indices, labels=words_least, rotation=90)
        plt.xlabel("Vārdi")
        plt.ylabel("Vārdu skaits")

        # Histogram plotting - Quote length histogram
        quote_len, quote_len_count = zip(*quote_len_hist)
        quote_len_count = np.array(list(quote_len_count))
        indices = np.arange(len(quote_len))
        plt.figure(figsize=(16.0, 9.0))
        plt.title("Citātu skaits atkarībā no garuma")
        plt.bar(indices, quote_len_count)
        plt.xticks(indices, quote_len, rotation=90)
        plt.xlabel("Citātu garumi")
        plt.ylabel("Citātu skaits")

        plt.ion()
        plt.show()
        del words, w_count, quote_len, quote_len_count, quote_len_hist, indices

