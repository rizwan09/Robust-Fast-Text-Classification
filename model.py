###############################################################################
# Author: Wasi Ahmad
# Project: Biattentive  Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script contains code related to the sequence-to-sequence
# network.
###############################################################################

import torch, helper
import torch.nn as nn
import torch.nn.functional as f
from collections import OrderedDict
from nn_layer import EmbeddingLayer, Encoder, MaxoutNetwork


# details of BCN can be found in the paper, "Learned in Translation: Contextualized Word Vectors"
class BCN(nn.Module):
    """Biattentive classification network architecture for sentence classification."""

    def __init__(self, dictionary, embedding_index, args):
        """"Constructor of the class."""
        super(BCN, self).__init__()
        self.config = args
        self.num_directions = 2 if self.config.bidirection else 1
        self.dictionary = dictionary

        self.embedding = EmbeddingLayer(len(self.dictionary), self.config.emsize, self.config.emtraining, self.config)
        self.embedding.init_embedding_weights(self.dictionary, embedding_index, self.config.emsize)

        self.relu_network = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(self.config.emsize, self.config.nhid)),
            ('nonlinearity', nn.ReLU())
        ]))

        self.encoder = Encoder(self.config.nhid, self.config.nhid, self.config.bidirection, self.config.nlayers,
                               self.config)
        self.biatt_encoder1 = Encoder(self.config.nhid * self.num_directions * 3, self.config.nhid, self.config.bidirection, 1,
                                      self.config)
        self.biatt_encoder2 = Encoder(self.config.nhid * self.num_directions * 3, self.config.nhid, self.config.bidirection, 1,
                                      self.config)

        self.ffnn = nn.Linear(self.config.nhid * self.num_directions, 1)
        self.maxout_network = MaxoutNetwork(self.config.nhid * self.num_directions * 4 * 2, self.config.num_class,
                                            num_units=self.config.num_units)

    def forward(self, sentence1, sentence1_len, sentence2, sentence2_len):
        """
        Forward computation of the biattentive classification network.
        Returns classification scores for a batch of sentence pairs.
        :param sentence1: 2d tensor [batch_size x max_length]
        :param sentence1_len: 1d numpy array [batch_size]
        :param sentence2: 2d tensor [batch_size x max_length]
        :param sentence2_len: 1d numpy array [batch_size]
        :return: classification scores over batch [batch_size x num_classes]
        """
        # step1: embed the words into vectors [batch_size x max_length x emsize]
        embedded_x = self.embedding(sentence1)
        embedded_y = self.embedding(sentence2)

        # step2: pass the embedded words through the ReLU network [batch_size x max_length x hidden_size]
        embedded_x = self.relu_network(embedded_x)
        embedded_y = self.relu_network(embedded_y)

        # step3: pass the word vectors through the encoder [batch_size x max_length x hidden_size * num_directions]
        encoded_x = self.encoder(embedded_x, sentence1_len)
        # For the second sentences in batch
        encoded_y = self.encoder(embedded_y, sentence2_len)

        # step4: compute affinity matrix [batch_size x sent1_max_length x sent2_max_length]
        affinity_mat = torch.bmm(encoded_x, encoded_y.transpose(1, 2))

        # step5: compute conditioned representations [batch_size x max_length x hidden_size * num_directions]
        conditioned_x = torch.bmm(f.softmax(affinity_mat, 2).transpose(1, 2), encoded_x)
        conditioned_y = torch.bmm(f.softmax(affinity_mat.transpose(1, 2), 2).transpose(1, 2), encoded_y)

        # step6: generate input of the biattentive encoders [batch_size x max_length x hidden_size * num_directions * 3]
        biatt_input_x = torch.cat(
            (encoded_x, torch.abs(encoded_x - conditioned_y), torch.mul(encoded_x, conditioned_y)), 2)
        biatt_input_y = torch.cat(
            (encoded_y, torch.abs(encoded_y - conditioned_x), torch.mul(encoded_y, conditioned_x)), 2)

        # step7: pass the conditioned information through the biattentive encoders
        # [batch_size x max_length x hidden_size * num_directions]
        biatt_x = self.biatt_encoder1(biatt_input_x, sentence1_len)
        biatt_y = self.biatt_encoder2(biatt_input_y, sentence2_len)

        # step8: compute self-attentive pooling features
        att_weights_x = self.ffnn(biatt_x.view(-1, biatt_x.size(2))).squeeze(1)
        att_weights_x = f.softmax(att_weights_x.view(*biatt_x.size()[:-1]), 1)
        att_weights_y = self.ffnn(biatt_y.view(-1, biatt_y.size(2))).squeeze(1)
        att_weights_y = f.softmax(att_weights_y.view(*biatt_y.size()[:-1]), 1)
        self_att_x = torch.bmm(biatt_x.transpose(1, 2), att_weights_x.unsqueeze(2)).squeeze(2)
        self_att_y = torch.bmm(biatt_y.transpose(1, 2), att_weights_y.unsqueeze(2)).squeeze(2)

        # step9: compute the joint representations [batch_size x hidden_size * num_directions * 4]
        # print (' self_att_x size: ', self_att_x.size())
        pooled_x = torch.cat((biatt_x.max(1)[0], biatt_x.mean(1), biatt_x.min(1)[0], self_att_x), 1)
        pooled_y = torch.cat((biatt_y.max(1)[0], biatt_y.mean(1), biatt_y.min(1)[0], self_att_y), 1)

        # step10: pass the pooled representations through the maxout network
        score = self.maxout_network(torch.cat((pooled_x, pooled_y), 1))
        return score