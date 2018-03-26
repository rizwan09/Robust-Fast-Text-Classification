###############################################################################
# Author: Wasi Ahmad
# Project: Biattentive Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script contains all the command line arguments.
###############################################################################

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='multitask neural session relevance framework')
    parser.add_argument('--data', type=str, default='../data/',
                        help='location of the data corpus')
    parser.add_argument('--task', type=str, default='IMDB',
                        help='name of the task [any one of snli, multinli, allnli, quora]')
    parser.add_argument('--test', type=str, default='test',
                        help='data partition on which test performance should be measured')
    parser.add_argument("--num_class", type=int, default=2,
                        help="number of classes in the multi-class classification")
    parser.add_argument('--max_words', type=int, default=-1,
                        help='maximum number of words (top ones) to be added to dictionary)')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_Tanh, RNN_RELU, LSTM, GRU)')
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="adam or sgd,lr=0.1")
    parser.add_argument("--lrshrink", type=float, default=5,
                        help="shrink factor for sgd")
    parser.add_argument("--minlr", type=float, default=1e-5,
                        help="minimum lr")
    parser.add_argument('--bidirection', action='store_false',
                        help='use bidirectional recurrent unit')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--emtraining', action='store_true',
                        help='train embedding layer')
    parser.add_argument("--ffnn_dim", type=int, default=300,
                        help="number of hidden units per fc layer")
    parser.add_argument("--num_units", type=int, default=5,
                        help="width of a unit in each layer/depth in fc, maxout network ")
    parser.add_argument("--nhid", type=int, default=300,
                        help="number of hidden units in RNN encoder")
    parser.add_argument('--lr_decay', type=float, default=.99,
                        help='decay ratio for learning rate')
    parser.add_argument('--max_example', type=int, default=-1,
                        help='number of training examples (-1 = all examples)')
    parser.add_argument('--tokenize', action='store_true',
                        help='tokenize instances using word_tokenize')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--max_norm', type=float, default=5.0,
                        help='gradient clipping')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='sparsity')
    parser.add_argument('--coherent', type=float, default=2.0,
                        help='coherent')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper limit of epoch')
    parser.add_argument('--early_stop', type=int, default=3,
                        help='early stopping criterion')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--gpu', type=int, default=2, metavar='N',
                        help='gpu index')
    parser.add_argument('--eval_batch_size', type=int, default=256, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed for reproducibility')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA for computation')
    parser.add_argument('--SAG', action='store_true',
                        help='Full enc type of classifier')
    parser.add_argument('--WAG', action='store_true',
                        help='Full enc type of classifier')
    parser.add_argument('--debug', action='store_true',
                        help='in debug mode or not')
    parser.add_argument('--print_every', type=int, default=100,
                        help='training report interval')
    parser.add_argument('--plot_every', type=int, default=100,
                        help='plotting interval')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='resume from last checkpoint (default: none)')
    parser.add_argument('--save_path', type=str, default='../bcn_output/',
                        help='path to save the final model')
    parser.add_argument('--word_vectors_file', type=str, default='glove.6B.300d.txt',
                        help='GloVe word embedding file name')
    parser.add_argument('--word_vectors_directory', type=str, default='../glove/',
                        help='Path of GloVe word embeddings')
    parser.add_argument('--load_classifier', type=str, default='../bcn_output/IMDB/full_ori.pth.tar',
                        help='Path of GloVe word embeddings')
    parser.add_argument('--load_selector', type=str, default='../bcn_output/IMDB/full_ori.pth.tar',
                        help='Path of GloVe word embeddings')
    parser.add_argument('--full_enc', type=int, default=1,
                        help='is to test with loading with full enc model')

    args = parser.parse_args()
    return args
