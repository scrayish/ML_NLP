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
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utility import Utility as util
import dataset


def main():
    # Argparse initialization:
    parser = argparse.ArgumentParser(description='Hyper parameters for model', add_help=True)
    parser.add_argument('-M', '--model', default="LSTMSimple", type=str,
                        help='Choose a model for usage (default = LSTMSimple)')
    parser.add_argument('-nd', '--make_new_datafile', default=False, type=bool,
                        help='If want to create a new data file, pass "True"')
    parser.add_argument('-pf', '--path_dataset_full', type=str,
                        default='C:/Users/matis/Documents/NLP_Files/Datafiles/quotes.json',
                        help='Pass filepath to full dataset, if passed, will automatically process data for usage')
    parser.add_argument('-pr', '--path_dataset_processed', type=str,
                        default='C:/Users/matis/Documents/NLP_Files/Datafiles/all_data.json',
                        help='Give path to already processed data file, if no file, will create new in that path')
    parser.add_argument('-pwt', '--path_weight_pretrained', type=str,
                        default='C:/Users/matis/Documents/NLP_Files/Model_Weights/LSTMSimple_PreTrained.tar',
                        help='Path for using pre-trained weights, if no pre-trained, then save new weights to path')
    parser.add_argument('-lg', '--path_tbx_logs', default=None, type=str,
                        help='Path for saving tensorboardX logs, if not given will use default parameter')
    parser.add_argument('-mh', '--need_hist', default=False, type=bool,
                        help='If need histogram of words and sentence length')
    parser.add_argument('-ep', '--epochs', default=50, type=int,
                        help='choose for how many epochs to train model (default = 50)')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                        help='set a learning rate for model (default = 1e-3)')
    parser.add_argument('-bs', '--batch_size', default=32, type=int,
                        help='set the size for batches (default = 32)')
    parser.add_argument('-hs', '--hidden_size', default=128, type=int,
                        help='set the hidden size (default = 128)')
    parser.add_argument('-ed', '--embedding_dims', default=32, type=int,
                        help='set dimensions for embeddings (default = 32)')
    parser.add_argument('-ly', '--layers', default=1, type=int,
                        help='set number of stacked cells in model (default=1)')

    args = parser.parse_args([])

    # Setting up device:
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Dataset formation
    dataset_train, dataset_test, vocabulary, word_count, total_word_count, end_token = dataset.form_dataset(
        create_new=args.make_new_datafile,
        path_full=args.path_dataset_full,
        path_processed=args.path_dataset_processed
    )

    # Hyper-parameters
    if args.path_tbx_logs is None:
        writer = SummaryWriter()
    else:
        writer = SummaryWriter(logdir=args.path_tbx_logs)
    epochs = args.epochs
    epoch = 0
    Model = getattr(__import__('models.' + args.model, fromlist=['Model']), 'Model')
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    if args.model == "LSTMBinary":
        layers = args.layers * 2
    else:
        layers = args.layers
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
    weight_coefficients = util().data_weight_coefficients(word_count, total_word_count)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=util().collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=util().collate_fn)

    model = Model(args, end_token)
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

    # Lists for loss and accuracy
    train_loss_total = []
    test_loss_total = []
    train_acc_total = []
    test_acc_total = []

    model_work_modes = ['train', 'test']

    plt.ion()
    plt.show()

    # Loop
    for epoch in tqdm(range(epoch, epochs, 1), desc='training network'):

        for meter in param_dict:
            param_dict[meter].reset()

        for data_set in dataloader_train, dataloader_test:

            for x, y, x_len in tqdm(data_set, desc='doing batches'):
                if data_set is dataloader_train:
                    torch.set_grad_enabled(True)
                    mode = model_work_modes[0]
                else:
                    torch.set_grad_enabled(False)
                    mode = model_work_modes[1]

                h_s = (torch.zeros(size=(layers, len(x), hidden_size)).to(device),
                       torch.zeros(size=(layers, len(x), hidden_size)).to(device))

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
                tmp = torch.arange(end_token).reshape(1, end_token).to(device)
                #  VVV == Such a hack - need explanation on this one, captain!
                y_target = (y_target == tmp).float()   # one hot encoded 0.0 or 1.0

                loss = torch.mean(-torch.sum(weight_coefficients * y_target * torch.log(y_prim_padded + 1e-16), dim=1))

                if data_set is dataloader_train:
                    loss.backward()
                    optimizer.step()
                    model.zero_grad()

                param_dict[f'{mode}_loss'].add(loss.to('cpu').item())
                param_dict[f'{mode}_acc'].add(
                    util().f1score(y_target.detach().to('cpu'), y_prim_padded.detach().to('cpu'))
                )

        # Model saving check:
        mode = model_work_modes[1]
        if param_dict[f'{mode}_loss'].value()[0] < loss_best:
            state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'loss': param_dict[f'{mode}_loss'].value()[0]
            }
            torch.save(state, filepath)
            loss_best = param_dict[f'{mode}_loss'].value()[0]

        for mode in model_work_modes:
            writer.add_scalar(f'{mode} loss', param_dict[f'{mode}_loss'].value()[0], global_step=epoch + 1)
            writer.add_scalar(f'{mode} accuracy', param_dict[f'{mode}_acc'].value()[0], global_step=epoch + 1)


        # Rollout
        torch.set_grad_enabled(False)
        y_done = []
        y_prim = []
        y_sentence = []
        rollout_sentence = []
        # Using only 1 word, so batch size is 1
        h_s = (torch.zeros(size=(1, 1, hidden_size)).to('cpu'),
               torch.zeros(size=(1, 1, hidden_size)).to('cpu'))
        # Obtain 1st word from sample sentence (just one)
        y_t = x[-1][0]
        y_sentence.append(y_t.data.numpy().tolist())
        y_t = torch.nn.utils.rnn.pack_sequence(y_t.reshape(shape=(1, 1)).to('cpu'))
        y_prim.append(y_t)
        for _ in range(25):
            y_t, h_s = model.forward(y_prim[-1], h_s)

            y_t = y_t.data.argmax()
            y_sentence.append(y_t.data.numpy().tolist())
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
        writer.add_text(tag='Rollout sentence', text_string=rollout_string, global_step=epoch + 1)

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
