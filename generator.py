"""

Quote generator platform

Instructions:

1) You need to have pre-processed data file for this and model weights, if you don't have, train some
and pass them as arguments for the generator.

2) When you run generator, input a word or a sequence of words (english only)

3) Input a number of how many words you want to generate for quote (excluding input)

4) You will receive generated quote.



NOTE: Model uses deep learning and has some limitations!

If you pass the same word over and over again, it will generate the same quote over and over again.
That's not a bug, that's a limitation for model, as it has learned to generate that specific quote.
To vary quotes, put in more words as input so you can get more variety.


"""

from __future__ import print_function
import argparse
import torch
from torchnlp.word_to_vector import GloVe
from pre_processing import DataProcessor as dp
from utility import Utility as util


def main():

    parser = argparse.ArgumentParser(description='Parameters for loading model for inference', add_help=True)
    parser.add_argument('-M', '--model', type=str, required=True, help='Write model name used for inference')
    parser.add_argument('-pwt', '--path_weights_pretrained', type=str, required=True, help='Give path to model weights')
    parser.add_argument('-pr', '--path_dataset_processed', type=str, required=True,
                        help='Load preprocessed data file')
    parser.add_argument('-hs', '--hidden_size', default=512, type=int,
                        help='Give hidden size which matches model hidden size (default = 512)')

    args, other_args = parser.parse_known_args()

    Model = getattr(__import__('models.' + args.model, fromlist=['Model']), 'Model')

    device = 'cpu'

    # Retrieve all things from prerocessed datafile
    all_quotes, vocabulary, \
    word_count, total_word_count, \
    end_token, quote_count = dp().open_preprocessed(path_data_processed=args.path_dataset_processed)
    # Retrieve embeddings
    glove = GloVe('6B')
    embeddings = []
    for word in word_count.keys():
        embeddings.append(glove[word])

    # Initialize model
    model = Model(args, end_token, embeddings).to(device=device)

    # Load model state dictionary:
    state = torch.load(args.path_weights_pretrained, map_location=device)
    model.load_state_dict(state['model_state'])

    # Loop for generation, generate to your hearts content!
    generate = True

    while generate:
        print("Input the starting word or sequence")
        input_data = input()
        print("Input the length of quote to be generated (excluding start input)")
        length = input()
        length = int(length)
        seq_length = len(input_data.split())
        input_data = util.words_to_label(input_data, vocabulary, rollout=True)

        # Rollout
        torch.set_grad_enabled(False)
        y_prim = []
        y_sentence = []
        rollout_sentence = []
        # Using only 1 word, so batch size is 1
        # h_s = (torch.zeros(size=(layers, 1, hidden_size)).to(device),
        # torch.zeros(size=(layers, 1, hidden_size)).to(device))
        h_s = None

        # Semantics check: if received a starting sequence longer than 1, manage that beforehand
        if seq_length > 1:
            for fragment in input_data:
                y_t = fragment
                y_sentence.append(y_t.data.numpy())
                y_t = torch.nn.utils.rnn.pack_sequence(y_t.reshape(shape=(1, 1)))
                y_t, h_s = model.forward(y_t, h_s)

                # Since passed multiple words, but need only last output and last hidden,
                # we pass only last output and take last hidden with us to model
                y_t = y_t.data[-1].argmax()
                if fragment == input_data[-1]:
                    y_t = torch.nn.utils.rnn.pack_sequence(y_t.reshape(shape=(1, 1)))
                    y_prim.append(y_t)
        else:
            y_t = input_data
            y_sentence.append(y_t.data.numpy())
            y_t = torch.nn.utils.rnn.pack_sequence(y_t.reshape(shape=(1, 1)))
            y_prim.append(y_t)

        for _ in range(length):
            y_t, h_s = model.forward(y_prim[-1], h_s)

            y_t = y_t.data.argmax()
            y_sentence.append(y_t.data.numpy())
            if y_t == end_token:
                break

            y_t = torch.nn.utils.rnn.pack_sequence(y_t.reshape(shape=(1, 1)))
            y_prim.append(y_t)

        for label in y_sentence:
            for key in vocabulary:
                if vocabulary[key] == label:
                    rollout_sentence.append(key)
                    break

        rollout_string = ' '.join(rollout_sentence)
        print("Generated sequence:")
        print(rollout_string)

        print()
        print("Generate a new quote? [y/n]")
        response = input()
        if response == 'n' or response == 'N':
            generate = False


if __name__ == '__main__':
    main()
