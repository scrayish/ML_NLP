"""

File for dataset class


"""
import torch
from torch.utils.data import Dataset
from pre_processing import DataProcessor as dp
from utility import Utility as util
from torchnlp.word_to_vector import GloVe


# Define dataset class:
class QuoteDataset(Dataset):
    def __init__(self, data, end_token):
        self.end_token = end_token
        self.data = [self.prep_data(s, self.end_token) for s in data]

    @staticmethod  # Prepares data
    def prep_data(s, end_token):
        x = torch.LongTensor(s)
        y = torch.roll(x, shifts=-1, dims=0)
        y[-1] = end_token
        x_len = len(s)
        return x, y, x_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def form_dataset(create_new, path_full, path_processed, data_range=0):
    # Preprocess data if needed, else open processed file
    if create_new:
        all_quotes, vocabulary,\
        word_count, total_word_count,\
        end_token, quote_count = dp().preprocess(path_data_full=path_full,
                                                 path_data_processed=path_processed, data_range=data_range)
    else:
        all_quotes, vocabulary, \
        word_count, total_word_count, \
        end_token, quote_count = dp().open_preprocessed(path_data_processed=path_processed)

    # Create datasets and prepare embeddings:
    glove = GloVe('6B')
    # Get embeddings
    embeddings = []
    for word in word_count.keys():
        embeddings.append(glove[word])
    all_quotes = util().words_to_label(all_quotes, vocabulary)
    x_data = all_quotes[:int(len(all_quotes) * 0.8)]
    y_data = all_quotes[int(len(all_quotes) * 0.8):]
    dataset_train = QuoteDataset(x_data, end_token)
    dataset_test = QuoteDataset(y_data, end_token)
    return dataset_train, dataset_test, vocabulary, word_count, total_word_count, end_token, quote_count, embeddings
