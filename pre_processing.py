"""

Functions for pre-processing data for project

"""

from __future__ import print_function
import re
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter
from langid import langid
from torchnlp.word_to_vector import GloVe


class DataProcessor(object):
    def __init__(self):
        super(DataProcessor, self).__init__()
        self.all_quotes: list = None
        self.vocabulary: dict = None
        self.word_count: dict = None
        self.total_word_count: int
        self.end_token: int
        self.quote_count: int


    def preprocess(self, path_data_full, path_data_processed, data_range):
        all_quotes = []
        words_hist = []
        glove = GloVe('6B')
        all_tokens = list(glove.token_to_index.keys())
        embeddings_pretrained = []
        # Different punctuations
        punct_dict = {
            '.': ' ', '!': ' ', '?': ' ', '...': ' ', ',': ' ', ';': ' ', ':': ' ', '\u201D': ' ', '\u2019\u2019': ' ',
            ' \'': ' ', '\' ': ' ', '  ': ' ', '   ': ' ', '"': ' ', '--': ' ', '-': ' ', '\u201C': '', '\u2019': ' ',
            '\u2026': ' ', '(': ' ', ')': ' ', '[': ' ', ']': ' ', '{': ' ', '}': ' ', '\u2014': ' ', '+': ' ',
            '„': ' ', '[]': ' ', '()': ' ', '{}': ' ', '=': ' ', '♕': ' ', '@': ' ', '*': ' ', '&': ' and ', '#': ' ',
            '~': ' ', '\u2E2E': ' ', '\u2640': ' ', '\\': ' ', '/': ' ', '\u2665': ' ', '\u2764': ' ', '\u2018': ' ',
            '\u265B': ' ', '\u262F': ' ', '\u2013': ' ', '\uFF07': ' ', '\uFF07\uFF07': ' ', '\uFF40': ' ',
            '\u02CB': ' ', '\u0300': ' ', '%': ' %', '\u02BC': ' ', '\u02BC\u02BC': ' ', 'ღ': ' ', '\u2500': ' ',
            '\u202c': ' ', '\u0301': ' ', '\u202A': ' ', '<': ' ', '>': ' ', '❞': ' ', 'ε': ' ', '\u2637': ' ',
            '↺': ' ', '®': ' ', '$': ' ', '❣': ' ', '\u2015': ' ', '\u0313': ' ', '\u201B': ' ', '\u2032': ' ',
            '\u05F3': ' ', '\'': ' ', '`': ' ', '\u200E': ' ',
        }
        # Word combinations and "shortcuts"
        short_dict = {
            'i\'m ': ' i am ', 'i\u0301m ': ' i am ', 'i\u2019m ': ' i am ', 'it\'s ': ' it is ', 'it\u2019s ': ' it is ',
            'it´s ': ' it is ', '\u00B4ll': ' will ', 'won\u00b4t ': ' will not ', '\u00B4re ': ' are ',
            '\u00B4ve ': ' have ', 'i\u00B4m ': ' i am ', ' won\'t ': ' will not ', 'i\u0060m ': ' i am ',
            'man\'s ': ' mans ', 'won\u2019t ': ' will not ', 'can\'t ': ' cannot ', '\'re ': ' are ',
            'can\u0060t ': ' cannot ', '\u0060ve ': ' have ', 'won\u0060t ': ' will not ', 'n\u0060t ': ' not ',
            '\u02B9s ': ' is ', '\u0374s ': ' is ', '\u0374ve ': ' have ', '\u0374re ': ' are ', '\u02B9ve ': ' have ',
            '\u02B9re ': ' are ', '\'ve ': ' have ', '\'ll ': ' will ', '\u0060ll ': ' will ', '\'d ': ' would ',
            'n\'t ': ' not ', '\'s ': ' is ', 'don\u2019t ': ' do not ', 'me\u2026 ': ' me ', '\u2019s ': ' is ',
            '\u2019re ': ' are ', '\u0060re ': ' are ', 'if\u2026 ': ' if ', 'day\u2026 ': ' day ', 'n\u2019t ': ' not ',
            '\u2019ll ': ' will ', '\u2019d ': ' would ', 'n´t ': ' not ', '\u0301re ': ' are ', '\u0301ve ': ' have ',
            '̵͇̿̿з ': ' ', '•̪ⓧ ': ' ', '̵͇̿̿ ': ' ', 'isno ': 'is no ', 'kissand ': 'kiss and', 'ryanlilly ': 'ryan lilly ',
            'meand ': 'me and', 'whatlooks ': 'what looks', 'girlfriendcut ': 'girlfriend cut', 'worldyou ': ' world you ',
            'heavenis ': ' heaven is ', 'worldso ': ' world so ', 'havebetter ': ' have better ',
            'unknownand ': ' unknown and ', ' allof ': ' all of ', ' tolook ': ' to look ', ' notaffect ': ' not affect ',
            'likea ': ' like a ', 'wantedas ': ' wanted as ', 'agonyof ': ' agony of ', 'skillthat ': ' skill that ',
            'worldsall ': ' worlds all ', 'awaywhat ': ' away what ', 'outwhat ': ' out what ', 'savewhat ': ' save what ',
            'educationso ': ' education so ', 'anyday ': ' any day ', 'usdo ': ' us do ',
            ' dependsupona ': ' depends upon a', ' wheelbarrowglazed ': ' wheelbarrow glazed ', 'waterbeside': 'water beside',
            ' whitechickens ': ' white chickens ',
        }
        with open(path_data_full, encoding='utf8') as json_file:
            data = json.load(json_file)
            if data_range == 0:
                data_range = len(data)
            for i in range(data_range):
                quote = data[i].get('Quote')

                # Detect language - if not english, then skip over
                # langID
                check_ld = langid.classify(quote)
                if check_ld[0] != 'en':
                    continue

                # Cleansing of quotes
                quote = re.sub(r"(?<![A-Z])(?<!^)([A-Z])", r" \1", quote)
                quote = quote.lower()
                quote = quote.strip('...')
                quote = quote.strip('"')
                quote = quote.strip()

                for diction in short_dict, punct_dict:
                    for inst in diction:
                        quote = quote.replace(inst, diction[inst])

                # Length control - if longer than 10 words, then fuck off mate
                enable_length_control = True
                if enable_length_control:
                    if len(quote.split()) > 5:
                        continue

                # Check if word is in tokens, if not, drop sentence:
                all_tokens_avail = True
                for word in quote.split():
                    if not word in all_tokens:
                        all_tokens_avail = False
                        break
                if not all_tokens_avail:
                    continue

                all_quotes.append(quote)
                for word in quote.split():
                    words_hist.append(word)

            self.total_word_count = len(words_hist)
            words_hist = Counter(words_hist)
            words_hist = words_hist.most_common(None)
            words, count = zip(*words_hist)
            self.word_count = dict(words_hist)
            vocabulary = dict(enumerate(words))
            self.end_token = len(vocabulary) + 1
            vocabulary[f'{self.end_token - 1}'] = self.end_token
            self.vocabulary = dict([(value, key) for key, value in vocabulary.items()])
            all_quotes = set(all_quotes)
            self.all_quotes = list(all_quotes)
            value = ''
            for i in self.all_quotes:
                if i == value:
                    self.all_quotes.remove(value)
                    break
            self.quote_count = len(self.all_quotes)

        json_file.close()

        # Create a list with all quotes
        processed_data = {
            'Quotes': self.all_quotes,
            'Vocabulary': self.vocabulary,
            'Word count': self.word_count,
            'Words total': self.total_word_count,
            'Quotes total': self.quote_count,
        }

        # Write to file:
        with open(path_data_processed, 'w') as outfile:
            json.dump(processed_data, outfile)

        outfile.close()
        # Delete all variables that are not needed anymore to clear memory
        del data, quote, words_hist, words, count, all_quotes, vocabulary
        # Return all needed values:
        return self.all_quotes, self.vocabulary, self.word_count,\
               self.total_word_count, self.end_token, self.quote_count,

    def open_preprocessed(self, path_data_processed):
        with open(path_data_processed, encoding='utf8') as json_file:
            data = json.load(json_file)
            self.all_quotes = data['Quotes']
            self.vocabulary = data['Vocabulary']
            self.word_count = data['Word count']
            self.total_word_count = data['Words total']
            self.quote_count = data['Quotes total']

        json_file.close()
        self.end_token = len(self.vocabulary)
        return self.all_quotes, self.vocabulary, self.word_count,\
               self.total_word_count, self.end_token, self.quote_count,

    def draw_histogram(self):
        matplotlib.use("TkAgg")
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
