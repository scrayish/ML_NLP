"""

Main file for calling models and using them as needed for

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
import models


def main():
    # Argparse initialization:
    parser = argparse.ArgumentParser(description='Hyper parameters for model', add_help=True)
    parser.add_argument('-M', '--model', default="Model_1", type=str, metavar='',
                        help='Choose 1 of 3 machine learning models for usage, syntax - Model_# (Default is Model_1)')
    parser.add_argument('-nd', '--make_new_datafile', default=False, type=bool, metavar='',
                        help='If want to create a new data file')
    parser.add_argument('-mh', '--need_hist', default=False, type=bool, metavar='',
                        help='If need histogram of words')
    parser.add_argument('-e', '--epochs', default=50, type=int, metavar='',
                        help='choose for how many epochs to train model')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, metavar='',
                        help='set a learning rate for model')
    parser.add_argument('-bs', '--batch_size', default=32, type=int, metavar='',
                        help='set the size for batches')
    parser.add_argument('-hs', '--hidden_size', default=128, type=int, metavar='',
                        help='set the hidden size')
    parser.add_argument('-ed', '--embedding_dims', default=32, type=int, metavar='',
                        help='set dimensions for embeddings')

    args = parser.parse_args()

    # Define dataset class:
    class QuoteDataset(Dataset):
        def __init__(self, data):
            self.data = [self.prep_data(s) for s in data]

        @staticmethod   # Prepares data
        def prep_data(s):
            x = torch.LongTensor(s)
            y = torch.roll(x, shifts=-1, dims=0)
            y[-1] = end_token
            x_len = len(s)
            return x, y, x_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]


    # All functions
    # OHE transform to label for f1_score
    def f1score(y, y_prim):
        y_true = np.zeros((len(y),))
        y_pred = np.zeros((len(y_prim),))

        for i in range(len(y_prim)):
            y_true[i] = y[i].argmax()
            y_pred[i] = y_prim[i].argmax()

        return np.mean(f1_score(y_true, y_pred, average='macro'))


    # Data labeling:
    def words_to_label(quotes):
        quotes_labeled = []
        for quote in tqdm(quotes, desc='labeling words in quotes'):
            words = quote.split()
            s_q_labeled = []

            for word in words:
                s_q_labeled.append(vocabulary[word])

            s_q_labeled = torch.LongTensor(np.asarray(s_q_labeled))
            quotes_labeled.append(s_q_labeled)

        return quotes_labeled


    # Function that can work with dynamic length sequences: (HP required above this function)
    def collate_fn(batch_input):
        x_input, y_input, x_len = zip(*batch_input)

        x_len_max = int(np.max(x_len))

        # sort batch so that max x_len first (descending)
        input_indexes_sorted = list(reversed(np.argsort(x_len).tolist()))

        #x = torch.zeros((batch_size, x_len_max), dtype=torch.long)
        #y = torch.zeros_like(x)

        x = []
        y = []

        x_len_out = torch.LongTensor(x_len)

        for i in range(len(batch_input)):
            i_sorted = input_indexes_sorted[i]
            x_len_out[i] = x_len[i_sorted]
            x.append(x_input[i_sorted])
            y.append(y_input[i_sorted])

            #x[i, 0:x_len_out[i]] = x_input[i_sorted]
            #y[i, 0:x_len_out[i]] = y_input[i_sorted]

        return (x, y, x_len_out)


    # Initialization to work without hitch
    # Data preparation - if file exists and don't want to make new, then open, else make new:
    data_fp = "C:/Users/matis/Documents/ML_prac/ML_gatavosanas/all_data.json"
    make_new = args.make_new_datafile
    fp = Path(data_fp)
    if fp.is_file() and make_new is False:   # Open existing:
        with open(data_fp, encoding='utf8') as json_file:
            data = json.load(json_file)
            all_quotes = data['Quotes']
            vocabulary = data['Vocabulary']
            word_count = data['Word count']
            total_word_count = data['Words total']
    else:   # Write again, can adjust parameters:
        all_quotes = []
        words_hist = []
        # Different punctuations
        punct_dict = {
            '.': '', '!': '', '?': '', '...': ' ', ',': ' ', ';': ' ', ':': ' ', '\u201D': ' ', '\u2019\u2019': ' ',
            ' \'': ' ', '\' ': ' ', '  ': ' ', '   ': ' ', '"': ' ', '--': ' ', '-': ' ', '\u201C': '', '\u2019': ' ',
            '\u2026': ' ', '(': ' ', ')': ' ', '[': ' ', ']': ' ', '{': ' ', '}': ' ', '\u2014': ' ', '+': ' ',
            '[]': ' ', '()': ' ', '{}': ' ', '=': ' ', '♕': ' ', '@': ' ', '*': ' ', '&': ' ', '#': ' ', '~': ' ',
            '\u2E2E': ' ', '\u2640': ' ', '\\': ' ', '/': ' ',  '\u2665': ' ', '\u2764': ' '
        }
        # Word combinations and "shortcuts"
        short_dict = {
            'i\'m': 'i am', 'i\u2019m': 'i am', 'it\'s': 'it is', 'it\u2019s': 'it is', 'won\'t': 'will not',
            'can\'t': 'cannot', '\'re': ' are', '\'ve': ' have', '\'ll': ' will', '\'d': ' would', 'n\'t': ' not',
            '\'s': ' is', 'don\u2019t': 'do not', 'me\u2026': 'me', '\u2019s': ' is', '\u2019re': ' are',
            'if\u2026': 'if ', 'day\u2026': 'day ', 'n\u2019t': ' not', '\u2019ll': ' will', '\u2019d': ' would'
        }
        with open('quotes.json', encoding='utf8') as json_file:
            data = json.load(json_file)
            for i in range(len(data)):
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

            total_word_count = len(words_hist)
            words_hist = Counter(words_hist)
            words_hist = words_hist.most_common(None)
            words, count = zip(*words_hist)
            word_count = dict(words_hist)
            vocabulary = dict(enumerate(words, 1))
            end_token = len(vocabulary) + 1
            vocabulary[f'{end_token}'] = end_token
            vocabulary = dict([(value, key) for key, value in vocabulary.items()])
            all_quotes = set(all_quotes)
            all_quotes = list(all_quotes)
            value = ''
            for i in all_quotes:
                if i == value:
                    all_quotes.remove(value)
                    break

        json_file.close()

        # Create a list with all quotes
        qoutes_and_vocab = {
            'Quotes': all_quotes,
            'Vocabulary': vocabulary,
            'Word count': word_count,
            'Words total': total_word_count
        }
        # Convert dict to JSON, so I can write in JSON file:
        #q_and_v_json = json.dumps(qoutes_and_vocab)
        # Write to file:
        with open('all_data.json', 'w') as outfile:
            json.dump(qoutes_and_vocab, outfile)

        outfile.close()

        # Open file for testing:
        with open(data_fp) as openfile:
            all_data = json.load(openfile)
            all_quotes = all_data['Quotes']
            vocabulary = all_data['Vocabulary']
            word_count = all_data['Word count']
            total_word_count = all_data['Words total']

        openfile.close()


    # Check if need histograms
    need_hist = args.need_hist
    if need_hist:
        # Count how many times each word shows up in quotes and draw histogram:
        percentile = 1
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
        w_count = w_count[:int(len(words) * (percentile / 100))]
        words = words[:int(len(words) * (percentile / 100))]
        # Sorts words by rarity (most common --> least common; num=how many examples)
        quote_len_hist = Counter(quote_len_hist)
        quote_len_hist = quote_len_hist.most_common(None)

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
        del words, w_count, quote_len, quote_len_count, quote_len_hist

    # Testing for google_colab and setting up device:
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False

    device = 'cpu'

    if IN_COLAB:
        device = 'cuda'


    # Hyper-parameters
    logdir = "C:/Users/matis/Documents/ML_prac/ML_gatavosanas/runs/QuoteGen"
    writer = SummaryWriter(logdir=logdir)
    epochs = args.epochs
    epoch = 0
    model = None
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    embedding_dims = args.embedding_dims
    vocab_len = len(vocabulary)
    end_token = int(vocabulary[f'{vocab_len}'])
    train_loss = tnt.meter.AverageValueMeter()
    test_loss = tnt.meter.AverageValueMeter()
    train_acc = tnt.meter.AverageValueMeter()
    test_acc = tnt.meter.AverageValueMeter()
    filepath = "C:/Users/matis/Documents/QuoteGen_Model_1.tar"
    param_dict = {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_acc': train_acc,
        'test_acc': test_acc
    }

    # Weight coefficient for weighted CELoss
    w_coefficient = []
    for i in word_count:
        w_coefficient.append(word_count[i])

    w_coefficient = np.array(w_coefficient)
    w_coefficient = 1 - w_coefficient / total_word_count
    w_coefficient = np.append(w_coefficient, 1 / int(total_word_count / 2))
    w_coefficient = torch.FloatTensor(w_coefficient).to(device)

    # Dataset formation
    all_quotes = words_to_label(all_quotes)
    x_data = all_quotes[:int(len(all_quotes) * 0.8)]
    y_data = all_quotes[int(len(all_quotes) * 0.8):]

    dataset_train = QuoteDataset(x_data)
    dataset_test = QuoteDataset(y_data)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model initialization, check if have pretrained weights, use if True:
    if args.model == "Model_1":
        model = models.Model1(batch_size=batch_size, hidden_size=hidden_size,
                              vocab_len=vocab_len, emb_dims=embedding_dims)
        filepath = "C:/Users/matis/Documents/QuoteGen_Model_1.tar"
    elif args.model == "Model_2":
        model = models.Model2(batch_size=batch_size, hidden_size=hidden_size,
                              vocab_len=vocab_len, emb_dims=embedding_dims).to(device)
        filepath = "C:/Users/matis/Documents/QuoteGen_Model_2.tar"
    elif args.model == "Model_3":
        model = models.Model3(batch_size=batch_size, hidden_size=hidden_size,
                              vocab_len=vocab_len, emb_dims=embedding_dims).to(device)
        filepath = "C:/Users/matis/Documents/QuoteGen_Model_3.tar"

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    fp = Path(filepath)
    IS_FILE = fp.is_file()
    loss_best = np.inf
    if IS_FILE:
        state = torch.load(filepath)
        epoch = state['epoch']
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optim_state'])
        loss_best = state['loss']
        model.train()
    else:
        # Initial save, so I get more consistency
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'loss': np.inf
        }
        torch.save(state, filepath)

    # Lists for loss and accuracy
    train_loss_total = []
    test_loss_total = []
    train_acc_total = []
    test_acc_total = []

    modes = ['train', 'test']

    if need_hist is False:
        plt.ion()
        plt.show()

    # Loop
    for epoch in tqdm(range(epochs), desc='training network'):

        for meter in param_dict:
            param_dict[meter].reset()

        for data_set in dataloader_train, dataloader_test:

            for x, y, x_len in tqdm(data_set, desc='doing batches'):
                if data_set is dataloader_train:
                    torch.set_grad_enabled(True)
                    mode = modes[0]
                else:
                    torch.set_grad_enabled(False)
                    mode = modes[1]

                h_s = (torch.zeros(size=(1, len(x), hidden_size)).to(device),
                       torch.zeros(size=(1, len(x), hidden_size)).to(device))

                x_pack = torch.nn.utils.rnn.pack_sequence(x)

                y_prim, h_s = model.forward(x_pack.to(device), h_s)

                y_prim_padded, len_out = torch.nn.utils.rnn.pad_packed_sequence(
                    y_prim,
                    batch_first=True,
                    total_length=x_len[0]
                )

                y_padded, len_out_y = torch.nn.utils.rnn.pad_packed_sequence(
                    torch.nn.utils.rnn.pack_sequence(y),
                    batch_first=True,
                    total_length=x_len[0]
                )

                y_prim_padded = y_prim_padded.contiguous().view((y_prim_padded.size(0) * y_prim_padded.size(1), -1))
                y_target = y_padded.contiguous().view((y_padded.size(0) * y_padded.size(1), 1)).to(device)
                # Removed .to(device), so this happens on CPU, then send all to GPU together
                tmp = torch.arange(vocab_len).reshape(1, vocab_len).to(device)
                #  VVV == Such a hack - need explanation on this one, captain!
                y_target = (y_target == tmp).float()   # one hot encoded 0.0 or 1.0

                loss = torch.mean(-torch.sum(w_coefficient * y_target * torch.log(y_prim_padded + 1e-16), dim=1))

                if data_set is dataloader_train:
                    loss.backward()
                    optimizer.step()
                    model.zero_grad()

                param_dict[f'{mode}_loss'].add(loss.to('cpu').item())
                param_dict[f'{mode}_acc'].add(f1score(y_target.detach().to('cpu'), y_prim_padded.detach().to('cpu')))

        # Model saving check:
        mode = modes[1]
        if param_dict[f'{mode}_loss'].value()[0] < loss_best:
            state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'loss': param_dict[f'{mode}_loss'].value()[0]
            }
            torch.save(state, filepath)
            loss_best = param_dict[f'{mode}_loss'].value()[0]

        for i in modes:
            writer.add_scalar(f'{i} loss', param_dict[f'{i}_loss'].value()[0], global_step=epoch + 1)
            writer.add_scalar(f'{i} accuracy', param_dict[f'{i}_acc'].value()[0], global_step=epoch + 1)

        # Rollout
        torch.set_grad_enabled(False)
        y_prim = []
        y_t = []
        h_s = (torch.zeros(size=(1, len(x), hidden_size)).to(device),
               torch.zeros(size=(1, len(x), hidden_size)).to(device))
        # Obtain 1st word from each sample
        for sample in x:
            y_t.append(torch.LongTensor(sample[0].reshape(1, )))

        y_t = torch.stack(y_t)
        # Pack y_t for rollout and do rollout:
        y_t = torch.nn.utils.rnn.pack_sequence(y_t)
        y_prim.append(y_t)
        for _ in range(25):
            y_t, h_s = model.forward(y_prim[-1].to(device), h_s)

            y_t_label = []
            for ohe in y_t.data:
                y_t_label.append(torch.LongTensor(ohe.argmax().reshape(1,)))

            y_t_label = torch.nn.utils.rnn.pack_sequence(y_t_label)
            y_prim.append(y_t_label)

        # Take results, make words happen:
        sentence_labels = []
        for batch in y_prim:
            sentence_labels.append(batch[0])

        sentence_words = []
        for label in sentence_labels:
            for key in vocabulary:
                if vocabulary[key] == label[0]:
                    sentence_words.append(key)
                    break

        sentence = ' '.join(sentence_words)
        print(sentence)

        train_loss_total.append(train_loss.value()[0])
        test_loss_total.append(test_loss.value()[0])
        train_acc_total.append(train_acc.value()[0])
        test_acc_total.append(test_acc.value()[0])

        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(epoch + 1), np.array(train_loss_total), 'r-', label='Train loss')
        plt.plot(np.arange(epoch + 1), np.array(test_loss_total), 'b-', label='Test loss')
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(epoch + 1), np.array(train_acc_total), 'g-', label='Train acc')
        plt.plot(np.arange(epoch + 1), np.array(test_acc_total), 'y-', label='Test acc')
        plt.legend()
        plt.grid()
        plt.draw()
        plt.pause(1e-2)

    plt.pause(0)


if __name__ == '__main__':
    main()
