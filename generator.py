"""

Quote generator platform

Instructions:

1) You need to have pre-processed data file for this and model weights, if you don't have, train some
and pass them as arguments for the generator.

2) When you run generator, input a word or a sequence of words (english only)

3) Input a number of how many words you want to generate for quote (excluding input)

4) You will receive generated quote.



NOTE: Model uses deep learning and has some limitations!

If you pass the same input over and over again, it will generate the same quote over and over again.
That's not a bug, that's a limitation for model, as it has learned to generate that specific quote.
To vary quotes, put in more words as input so you can get more variety.

12.06.2020. ^^^ This is no longer the case lads! All good!
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
    parser.add_argument('-sau', '--s_a_unit_count', required=False, type=int, default=5,
                        help="Self attention unit count for model (BuildBlocks only)")
    parser.add_argument('-lc', '--layer_count', required=False, type=int, default=2,
                        help="How many encoding/decoding layers (BuildBlocks only)")
    parser.add_argument('-d', '--dimensions', required=False, type=int, default=64,
                        help="Inner dimensions for Q, K, V Matrices (BuildBlocks only)")

    args, other_args = parser.parse_known_args()

    Model = getattr(__import__('models.' + args.model, fromlist=['Model']), 'Model')

    device = 'cpu'

    # If transformer spotted, use that:
    transformer = False
    if 'BuildBlocks' in args.model:
        transformer = True

    # Retrieve all things from prerocessed datafile
    all_quotes, vocabulary, \
    word_count, total_word_count, \
    end_token, quote_count = dp().open_preprocessed(path_data_processed=args.path_dataset_processed)

    # Check model for embedding usage determination:
    emb_phrase = 'Glove'
    if emb_phrase in args.model or 'BuildBlocks' in args.model:
        # Retrieve embeddings
        glove = GloVe('6B')
        embeddings = []
        for word in word_count.keys():
            embeddings.append(glove[word])

        # Initialize model
        model = Model(args, embeddings).to(device=device)
    else:
        model = Model(args, end_token).to(device=device)

    # Load model state dictionary and set up model for generating:
    state = torch.load(args.path_weights_pretrained, map_location=device)
    model.load_state_dict(state['model_state'])
    # To make use of Dropout layers, if set to eval(), dropout won't work.
    model.train()
    # Disable grad because not required
    torch.set_grad_enabled(False)

    # Loop for generation, generate to your hearts content!
    generate = True
    print("Quote Generator v1.0")
    print(f'Running model: {args.model}')

    while generate:
        print("Input the starting word or sequence")
        input_data = input()
        print("Input the length of quote to be generated (excluding start input)")
        length = input()
        length = int(length)
        if input_data == '':
            print("Empty input, defaulting to 'I'")
            input_data = 'I'
        input_data = input_data.lower()
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
            for i in range(len(input_data)):
                y_t = input_data[i]
                y_sentence.append(y_t.data.numpy())
                if transformer:

                    # Take random only if last element of list:
                    if input_data[i] == input_data[-1]:
                        y = torch.randint(low=0, high=end_token + 1, size=y_t.size())
                    else:
                        y = input_data[i + 1]

                    y_t = y_t.reshape(shape=(1, 1))
                    y_t = model.forward(y_t, y)
                else:
                    y_t = torch.nn.utils.rnn.pack_sequence(y_t.reshape(shape=(1, 1)))
                    y_t, h_s = model.forward(y_t, h_s)

                # Since passed multiple words, but need only last output and last hidden,
                # we pass only last output and take last hidden with us to model
                y_t = y_t.data[-1].argmax()
                if input_data[i] == input_data[-1]:
                    y_t = y_t.reshape(shape=(1, 1))
                    if not transformer:
                        y_t = torch.nn.utils.rnn.pack_sequence(y_t)
                    y_prim.append(y_t)
        else:
            y_t = input_data
            y_sentence.append(y_t.data.numpy())
            y_t = y_t.reshape(shape=(1, 1))
            if not transformer:
                y_t = torch.nn.utils.rnn.pack_sequence(y_t)
            y_prim.append(y_t)

        for _ in range(length):
            if transformer:
                y = torch.randint(low=0, high=end_token + 1, size=y_t.size())
                y_t = model.forward(y_prim[-1], y)
            else:
                y_t, h_s = model.forward(y_prim[-1], h_s)

            y_t = y_t.data.argmax()
            y_sentence.append(y_t.data.numpy())
            if y_t == end_token:
                break

            y_t = y_t.reshape(shape=(1, 1))
            if not transformer:
                y_t = torch.nn.utils.rnn.pack_sequence(y_t)
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

    exit()


if __name__ == '__main__':
    main()
