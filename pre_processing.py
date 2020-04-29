"""

Functions for pre-processing data for project

"""

from __future__ import print_function
import re
import argparse
from tensorboardX import SummaryWriter
from pathlib import Path
import json
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torchnet as tnt
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from collections import Counter


class DataProcessor(object):
    def __init__(self):
        super(DataProcessor, self).__init__()
        self.all_quotes: list = None
        self.vocabulary: dict = None
        self.word_count: dict = None
        self.total_word_count: int
        self.end_token: int

    def preprocess(self, path_data_full, path_data_processed, data_range=0):
        all_quotes = []
        words_hist = []
        # Different punctuations
        punct_dict = {
            '.': '', '!': '', '?': '', '...': ' ', ',': ' ', ';': ' ', ':': ' ', '\u201D': ' ', '\u2019\u2019': ' ',
            ' \'': ' ', '\' ': ' ', '  ': ' ', '   ': ' ', '"': ' ', '--': ' ', '-': ' ', '\u201C': '', '\u2019': ' ',
            '\u2026': ' ', '(': ' ', ')': ' ', '[': ' ', ']': ' ', '{': ' ', '}': ' ', '\u2014': ' ', '+': ' ',
            '[]': ' ', '()': ' ', '{}': ' ', '=': ' ', '♕': ' ', '@': ' ', '*': ' ', '&': ' ', '#': ' ', '~': ' ',
            '\u2E2E': ' ', '\u2640': ' ', '\\': ' ', '/': ' ', '\u2665': ' ', '\u2764': ' '
        }
        # Word combinations and "shortcuts"
        short_dict = {
            'i\'m': 'i am', 'i\u2019m': 'i am', 'it\'s': 'it is', 'it\u2019s': 'it is', 'won\'t': 'will not',
            'can\'t': 'cannot', '\'re': ' are', '\'ve': ' have', '\'ll': ' will', '\'d': ' would', 'n\'t': ' not',
            '\'s': ' is', 'don\u2019t': 'do not', 'me\u2026': 'me', '\u2019s': ' is', '\u2019re': ' are',
            'if\u2026': 'if ', 'day\u2026': 'day ', 'n\u2019t': ' not', '\u2019ll': ' will', '\u2019d': ' would'
        }
        with open(path_data_full, encoding='utf8') as json_file:
            data = json.load(json_file)
            if data_range == 0:
                data_range = len(data)
            for i in range(data_range):
                quote = data[i].get('Quote')

                # Cleansing of quotes
                quote = re.sub(r"(?<![A-Z])(?<!^)([A-Z])", r" \1", quote)
                quote = quote.lower()
                quote = quote.strip('...')
                quote = quote.strip('"')
                quote = quote.strip()

                for diction in short_dict, punct_dict:
                    for inst in diction:
                        quote = quote.replace(inst, diction[inst])

                all_quotes.append(quote)
                for word in quote.split():
                    words_hist.append(word)

            self.total_word_count = len(words_hist)
            words_hist = Counter(words_hist)
            words_hist = words_hist.most_common(None)
            words, count = zip(*words_hist)
            self.word_count = dict(words_hist)
            vocabulary = dict(enumerate(words, 1))
            self.end_token = len(vocabulary) + 1
            vocabulary[f'{self.end_token}'] = self.end_token
            self.vocabulary = dict([(value, key) for key, value in vocabulary.items()])
            all_quotes = set(all_quotes)
            self.all_quotes = list(all_quotes)
            value = ''
            for i in self.all_quotes:
                if i == value:
                    self.all_quotes.remove(value)
                    break

        json_file.close()

        # Create a list with all quotes
        processed_data = {
            'Quotes': self.all_quotes,
            'Vocabulary': self.vocabulary,
            'Word count': self.word_count,
            'Words total': self.total_word_count
        }
        # Convert dict to JSON, so I can write in JSON file:
        # q_and_v_json = json.dumps(qoutes_and_vocab)
        # Write to file:
        with open(path_data_processed, 'w') as outfile:
            json.dump(processed_data, outfile)

        outfile.close()
        # Delete all variables that are not needed anymore to clear memory
        del data, quote, words_hist, words, count, all_quotes, vocabulary
        # Return all needed values:
        return self.all_quotes, self.vocabulary, self.word_count, self.total_word_count, self.end_token

    def open_preprocessed(self, path_data_processed):
        with open(path_data_processed, encoding='utf8') as json_file:
            data = json.load(json_file)
            self.all_quotes = data['Quotes']
            self.vocabulary = data['Vocabulary']
            self.word_count = data['Word count']
            self.total_word_count = data['Words total']

        json_file.close()
        self.end_token = len(self.vocabulary)
        return self.all_quotes, self.vocabulary, self.word_count, self.total_word_count, self.end_token

    def draw_histogram(self):
        # Count how many times each word shows up in quotes and draw histogram:
        quote_len_hist = []
        for quote in self.all_quotes:
            quote_len_hist.append(len(quote.split()))

        words = []
        w_count = []
        for i in self.vocabulary:
            words.append(i)
        for i in self.word_count:
            w_count.append(self.word_count[i])
        words = np.array(words)
        w_count = np.array(w_count)
        w_count = w_count[:100]
        words = words[:100]
        quote_len_hist = Counter(quote_len_hist)

        # Splits off words from their count
        indices = np.arange(len(words))
        bar_width = 0.2
        # Histogram plotting - Words histogram
        plt.figure(figsize=(14.4, 9.0))
        plt.bar(indices, w_count)
        plt.xticks(ticks=indices, labels=words, rotation=90)
        plt.xlabel("words")
        plt.ylabel("count per word")

        # Histogram plotting - Quote length histogram
        quote_len, quote_len_count = zip(*quote_len_hist)
        quote_len_count = np.array(quote_len_count)
        indices = np.arange(len(quote_len))
        plt.figure(figsize=(14.4, 9.0))
        plt.bar(indices, quote_len_count)
        plt.xticks(indices, quote_len, rotation=90)
        plt.xlabel("length of quotes")
        plt.ylabel("quote count per length")

        plt.ion()
        plt.show()
        del words, w_count, quote_len, quote_len_count, quote_len_hist, indices
