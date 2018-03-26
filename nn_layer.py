###############################################################################
# Author: Wasi Ahmad
# Project: Biattentive Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script contains code related to the neural network layers.
###############################################################################

import torch, helper
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from collections import OrderedDict
from torch.autograd import Variable


class EmbeddingLayer(nn.Module):
    """Embedding class which includes only an embedding layer."""

    def __init__(self, input_size, emsize, emtraining, config):
        """"Constructor of the class"""
        super(EmbeddingLayer, self).__init__()

        if emtraining:
            self.embedding = nn.Sequential(OrderedDict([
                ('embedding', nn.Embedding(input_size, emsize)),
                ('dropout', nn.Dropout(config.dropout))
            ]))
        else:
            self.embedding = nn.Embedding(input_size, emsize)
            self.embedding.weight.requires_grad = False

    def forward(self, input_variable):
        """"Defines the forward computation of the embedding layer."""
        return self.embedding(input_variable)

    def init_embedding_weights(self, dictionary, embeddings_index, embedding_dim):
        """Initialize weight parameters for the embedding layer."""
        pretrained_weight = np.empty([len(dictionary), embedding_dim], dtype=float)
        for i in range(len(dictionary)):
            if dictionary.idx2word[i] in embeddings_index:
                pretrained_weight[i] = embeddings_index[dictionary.idx2word[i]]
            else:
                pretrained_weight[i] = helper.initialize_out_of_vocab_words(embedding_dim)
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        if isinstance(self.embedding, nn.Sequential):
            self.embedding[0].weight.data.copy_(torch.from_numpy(pretrained_weight))
        else:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))


class Encoder(nn.Module):
    """Encoder class of a sequence-to-sequence network"""

    def __init__(self, input_size, hidden_size, bidirection, nlayers, config):
        """"Constructor of the class"""
        super(Encoder, self).__init__()
        self.config = config
        if self.config.model in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.config.model)(input_size, hidden_size, nlayers, batch_first=True,
                                                      dropout=self.config.dropout, bidirectional=bidirection)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(input_size, hidden_size, nlayers, nonlinearity=nonlinearity, batch_first=True,
                              dropout=self.config.dropout, bidirectional=bidirection)

    def forward(self, sent_variable, sent_len):
        """"Defines the forward computation of the encoder"""
        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda() if self.config.cuda else torch.from_numpy(idx_sort)
        sent_variable = sent_variable.index_select(0, Variable(idx_sort))

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent_variable, sent_len, batch_first=True)
        sent_output = self.rnn(sent_packed)[0]
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.config.cuda else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(0, Variable(idx_unsort))

        return sent_output


class MaxoutNetwork(nn.Module):
    """3 layers Maxout network"""

    def __init__(self, nhid, num_classes, num_units=2):
        super(MaxoutNetwork, self).__init__()
        self.fc1_list = nn.ModuleList()
        self.fc2_list = nn.ModuleList()
        self.fc3_list = nn.ModuleList()
        self.batch_norm1 = nn.BatchNorm1d(nhid // 4)
        self.batch_norm2 = nn.BatchNorm1d(nhid // 8)
        for _ in range(num_units):
            self.fc1_list.append(nn.Linear(nhid, nhid // 4))
            self.fc2_list.append(nn.Linear(nhid // 4, nhid // 8))
            self.fc3_list.append(nn.Linear(nhid // 8, num_classes))

    def forward(self, x):
        x = f.dropout(x, p=0.1, training=self.training)
        x = self.batch_norm1(self.maxout(x, self.fc1_list))
        x = f.dropout(x, p=0.2, training=self.training)
        x = self.batch_norm2(self.maxout(x, self.fc2_list))
        x = f.dropout(x, p=0.3, training=self.training)
        x = self.maxout(x, self.fc3_list)
        return x

    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output


class WE_Selector(nn.Module):
    # selector selects the important words by probability output tensor
    def __init__(self, emsize, dropout, num_labels=1):
        super(WE_Selector, self).__init__()
        self.linear= nn.Linear(emsize, num_labels)
    

    #input (x) : (batch x sentence len x em_size)
    #output (x) : (batch x sentence len)
    def forward(self, x):
        score = self.linear(x).squeeze(2)
        # return pbz, f.log_softmax(score, dim=2)
        return f.sigmoid(score)

        
        