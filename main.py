"""

Main file for calling models and using them as needed for


"""

from __future__ import print_function
import argparse
from tensorboardX import SummaryWriter
from pathlib import Path
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torchnet as tnt
from torch.utils.data import DataLoader
from utility import Utility as util
import dataset


def main():
    # Argparse initialization:
    parser = argparse.ArgumentParser(description='Hyper parameters for models. For using RNN, pass hidden_size. for '
                                                 'using Transformer, pass layer_count, s_a_unit_count and dimensions.',
                                     add_help=True)
    parser.add_argument('-M', '--model', type=str, required=True, help='Choose a model for usage')
    parser.add_argument('-nd', '--make_new_datafile', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='If want to create a new data file, pass "True"')
    parser.add_argument('-dr', '--data_range', default=0, type=int, required=False,
                        help='For processing data set range, if \'0\', then will default to full data range')
    parser.add_argument('-pf', '--path_dataset_full', type=str, required=True,
                        help='Pass filepath to full dataset, if passed, will automatically process data for usage')
    parser.add_argument('-pr', '--path_dataset_processed', type=str, required=True,
                        help='Give path to already processed data file, if no file, will create new in that path')
    parser.add_argument('-sw', '--save_weights', default=False, type=bool,
                        help='Save pre-trained weights, pass True if want to save (default is False)')
    parser.add_argument('-pwt', '--path_weight_pretrained', type=str, required=True,
                        help='Path for using pre-trained weights, if no pre-trained, then save new weights to path')
    parser.add_argument('-lg', '--path_tbx_logs', default=None, type=str, required=False,
                        help='Path for saving tensorboardX logs, if not given will use default location')
    parser.add_argument('-mh', '--need_hist', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='If need histogram of words and sentence length')
    parser.add_argument('-ep', '--epochs', default=50, type=int,
                        help='choose for how many epochs to train model (default = 50)')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                        help='set a learning rate for model (default = 1e-3)')
    parser.add_argument('-bs', '--batch_size', default=32, type=int,
                        help='set the size for batches (default = 32)')
    parser.add_argument('-hs', '--hidden_size', default=128, type=int, required=False,
                        help='set the hidden size (RNN only)')
    parser.add_argument('-sau', '--s_a_unit_count', required=False, type=int, default=5,
                        help="Self attention unit count for model (Transformer only)")
    parser.add_argument('-lc', '--layer_count', required=False, type=int, default=2,
                        help="How many encoding/decoding layers (Transformer only)")
    parser.add_argument('-d', '--dimensions', required=False, type=int, default=64,
                        help="Inner dimensions for Q, K, V Matrices (Transformer only)")

    args, args_other = parser.parse_known_args()

    # Setting up device:
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Determine if need to use GloVe embeddings:
    emb_phrase = 'Glove'
    if emb_phrase in args.model or 'Transformer' in args.model:
        use_glove = True
    else:
        use_glove = False

    # Dataset formation
    dataset_train, dataset_test, vocabulary, word_count,\
    total_word_count, end_token, quote_count, embeddings = dataset.form_dataset(
        create_new=args.make_new_datafile,
        path_full=args.path_dataset_full,
        path_processed=args.path_dataset_processed,
        need_hist=args.need_hist,
        use_glove=use_glove,
        data_range=args.data_range,
    )

    # Hyper-parameters
    comment = f'LR_{args.learning_rate}_BATCH_{args.batch_size}_HIDDEN_{args.hidden_size}'
    if args.path_tbx_logs is None:
        writer = SummaryWriter(comment=comment)
    else:
        writer = SummaryWriter(logdir=args.path_tbx_logs, comment=comment)

    epochs = args.epochs
    epoch = 0
    Model = getattr(__import__('models.' + args.model, fromlist=['Model']), 'Model')
    batch_size = args.batch_size

    train_loss = tnt.meter.AverageValueMeter()
    test_loss = tnt.meter.AverageValueMeter()
    train_acc = tnt.meter.AverageValueMeter()
    test_acc = tnt.meter.AverageValueMeter()
    param_dict = {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_acc': train_acc,
        'test_acc': test_acc
    }

    # Weight coefficient for weighted CELoss
    weight_coefficients = util.data_weights(word_count, total_word_count, quote_count)
    weight_coefficients = weight_coefficients.to(device)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=util.collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=util.collate_fn)

    # If embeddings are None, model doesn't need them, so initialize a bit differently
    if embeddings is None:
        model = Model(args, end_token).to(device=device)
    else:
        model = Model(args, embeddings).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    fp = Path(args.path_weight_pretrained)
    IS_FILE = fp.is_file()
    loss_best = np.inf
    if IS_FILE:
        state = torch.load(args.path_weight_pretrained)
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
        torch.save(state, args.path_weight_pretrained)

    model_work_modes = ['train', 'test']

    # Do a check for transformer or LSTM/GRU for modifications in training loop:
    if 'Transformer' in args.model:
        transformer = True
    else:
        transformer = False

    # Loop
    for epoch in tqdm(range(epoch, epochs, 1), desc='training network', ncols=100):

        for meter in param_dict:
            param_dict[meter].reset()

        for data_set in dataloader_train, dataloader_test:
            for x, y, x_len in tqdm(data_set, desc="Going through data"):
                if data_set is dataloader_train:
                    torch.set_grad_enabled(True)
                    mode = model_work_modes[0]
                else:
                    torch.set_grad_enabled(False)
                    mode = model_work_modes[1]

                # Setup training with Transformer/(LSTM/GRU):
                if transformer:
                    # Padding sequences:
                    x_padded = torch.nn.utils.rnn.pad_sequence(
                        sequences=x,
                        batch_first=True,
                        padding_value=0,
                    )
                    y_padded = torch.nn.utils.rnn.pad_sequence(
                        sequences=y,
                        batch_first=True,
                        padding_value=0,
                    )
                    y_prim = model.forward(x_padded.to(device), y_padded.to(device))

                else:
                    h_s = None
                    x_pack = torch.nn.utils.rnn.pack_sequence(x)

                    y_prim, h_s = model.forward(x_pack.to(device), h_s)

                    y_prim, len_out = torch.nn.utils.rnn.pad_packed_sequence(
                        y_prim,
                        batch_first=True,
                        total_length=x_len[0]
                    )

                    y_padded, len_out_y = torch.nn.utils.rnn.pad_packed_sequence(
                        torch.nn.utils.rnn.pack_sequence(y),
                        batch_first=True,
                        total_length=x_len[0]
                    )

                # Create y_target OHE vectors for loss calculation and contiguous variables to one memory space:
                y_prim = y_prim.contiguous().view((y_prim.size(0) * y_prim.size(1), -1))
                y_target = y_padded.contiguous().view((y_padded.size(0) * y_padded.size(1), 1)).to(device)
                tmp = torch.arange(end_token + 1).reshape(1, -1).to(device)
                #  VVV == Such a hack - need explanation on this one, captain!
                y_target = (y_target == tmp).float()   # one hot encoded 0.0 or 1.0

                # Calculate loss:
                loss = torch.mean(-torch.sum(weight_coefficients * y_target * torch.log(y_prim + 1e-16),
                                             dim=1))

                if data_set is dataloader_train:
                    loss.backward()
                    optimizer.step()
                    model.zero_grad()

                param_dict[f'{mode}_loss'].add(loss.to('cpu').item())
                # Calculate accuracy:
                if transformer:
                    param_dict[f'{mode}_acc'].add(
                        util.f1score(y_target.detach().to('cpu'), y_prim.detach().to('cpu'))
                    )
                else:
                    param_dict[f'{mode}_acc'].add(
                        util.f1score(y_target.detach().to('cpu'), y_prim.detach().to('cpu'))
                    )

        # Save loss/accuracy measures to writer:
        for mode in model_work_modes:
            writer.add_scalar(f'{mode} loss', param_dict[f'{mode}_loss'].value()[0], global_step=epoch + 1)
            writer.add_scalar(f'{mode} accuracy', param_dict[f'{mode}_acc'].value()[0], global_step=epoch + 1)

        # Rollout operation. Starting with default set parameters:
        # TODO: Re-define rollout operation so transformer can be rolled out as well. = DONE
        torch.set_grad_enabled(False)
        y_prim = []
        y_sentence = []
        rollout_sentence = []
        if transformer:
            # Get starting word and generate random output for transformer:
            # 1st attempt - generating random sequence for decoder input.
            y_t = x[-1][0]
            y_t = y_t.reshape(shape=(1, 1)).to(device)
            y_sentence.append(y_t.data.numpy().tolist())
            y_prim.append(y_t)
            # Generate rollout sequence by feeding network output as input:
            for _ in range(25):
                y = torch.randint(low=0, high=end_token + 1, size=y_t.size())
                y_t = model.forward(y_prim[-1], y)
                y_t = y_t.to('cpu')
                y_t = y_t.data.argmax()
                y_sentence.append(y_t.data.numpy().tolist())

                y_t = y_t.reshape(shape=(1, 1))
                y_prim.append(y_t.to(device))
            pass
        else:
            # Using only 1 word, so batch size is 1
            h_s = None
            # Obtain 1st word from sample sentence (just one)
            y_t = x[-1][0]
            y_sentence.append(y_t.data.numpy().tolist())
            y_t = torch.nn.utils.rnn.pack_sequence(y_t.reshape(shape=(1, 1)).to(device))
            y_prim.append(y_t)
            # Generate rollout sequence by feeding network output as input:
            for _ in range(25):
                y_t, h_s = model.forward(y_prim[-1], h_s)

                # Send output to cpu and transform to predicted label:
                y_t = y_t.to('cpu')
                y_t = y_t.data.argmax()
                y_sentence.append(y_t.data.numpy().tolist())
                # If network generates EOS token, break the loop:
                if y_t == end_token:
                    break

                y_t = torch.nn.utils.rnn.pack_sequence(y_t.reshape(shape=(1, 1)))
                y_prim.append(y_t.to(device))

        # Replace word labels with words from dictionary:
        for label in y_sentence:
            for key in vocabulary:
                if vocabulary[key] == label:
                    rollout_sentence.append(key)
                    break

        # Join words into one string and save to writer object as rollout result:
        rollout_string = ' '.join(rollout_sentence)
        writer.add_text(tag='Rollout sentence', text_string=rollout_string, global_step=epoch + 1)

        # Save weights after 30th epoch because shit weights have no meaning
        if epoch >= 30:
            state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'loss': param_dict['test_loss'].value()[0]
            }
            torch.save(state, args.path_weight_pretrained)


if __name__ == '__main__':
    main()
