import torch, helper
import torch.nn as nn
import torch.nn.functional as f
from collections import OrderedDict
from nn_layer import EmbeddingLayer, WE_Selector


# details of BCN can be found in the paper, "Learned in Translation: Contextualized Word Vectors"
class Selector(nn.Module):
    """Biattentive classification network architecture for sentence classification."""

    def __init__(self, dictionary, embedding_index, args):
        """"Constructor of the class."""

        super(Selector, self).__init__()
        self.config = args
        self.dictionary = dictionary
        self.embedding = EmbeddingLayer(len(self.dictionary), self.config.emsize, self.config.emtraining, self.config)
        self.embedding.init_embedding_weights(self.dictionary, embedding_index, self.config.emsize)

        self.we_selector = WE_Selector(self.config.emsize, self.config.dropout)


    def forward(self, sentence1, sentence1_len_old, sentence2, sentence2_len_old, threshold = 0.5, is_train = 0):
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
        embedded_x1 = self.embedding(sentence1)
        embedded_y1 = self.embedding(sentence2)

        
        ###################################### selection ######################################
        pbx = self.we_selector(embedded_x1)
        pby = self.we_selector(embedded_y1)

        assert pbx.size() == sentence1.size()
        assert pby.size() == sentence2.size()

        #torch byte tesnor Variable of size (batch x len)
        selection_x = pbx.bernoulli().long()#(pbx>=threshold).long()
        selection_y = pby.bernoulli().long()#(pby>=threshold).long() 

        result_x = sentence1.mul(selection_x) #word ids that are selected; contains zeros where it's not selected (ony selected can be found by selected_x[selected_x!=0])
        result_y = sentence2.mul(selection_y) 

        selected_x, sentence1_len = helper.get_selected_tensor(result_x, pbx, sentence1, sentence1_len_old, self.config.cuda) #sentence1_len is a numpy array
        selected_y, sentence2_len = helper.get_selected_tensor(result_y, pby, sentence2, sentence2_len_old, self.config.cuda) #sentence2_len is a numpy array

        logpz = zsum = zdiff = -1.0
        if is_train==1:
            mask1 = (sentence1!=0).long()
            mask2 = (sentence2!=0).long()

            masked_selection_x =  selection_x.mul(mask1)
            masked_selection_y =  selection_y.mul(mask2)

            #logpz (batch x len)
            logpx = -helper.binary_cross_entropy(pbx, selection_x.float().detach(), reduce = False) #as reduce is not available for this version I am doing this code myself:
            logpy = -helper.binary_cross_entropy(pby, selection_y.float().detach(), reduce = False)
            assert logpx.size()== sentence1.size()

            # batch
            logpx = logpx.mul(mask1.float()).sum(1)
            logpy = logpy.mul(mask2.float()).sum(1)
            logpz = (logpx+logpy) 
            # zsum = ##### same as sentence1_len #####T.sum(z, axis=0, dtype=theano.config.floatX) 
            zdiff1 = (masked_selection_x[:,1:]-masked_selection_x[:,:-1]).abs().sum(1)  ####T.sum(T.abs_(z[1:]-z[:-1]), axis=0, dtype=theano.config.floatX)
            zdiff2 = (masked_selection_y[:,1:]-masked_selection_y[:,:-1]).abs().sum(1)  ####T.sum(T.abs_(z[1:]-z[:-1]), axis=0, dtype=theano.config.floatX)
            
            assert zdiff1.size()[0] == sentence1.size()[0]
            assert logpz.size()[0] == sentence1.size()[0]

            zdiff = zdiff1+zdiff2

            xsum = masked_selection_x.sum(1)
            ysum = masked_selection_y.sum(1)
            zsum = xsum+ysum

            assert zsum.size()[0] ==  sentence1.size()[0]

            assert logpz.dim() == zsum.dim()
            assert logpz.dim() == zdiff.dim()
            return selected_x, sentence1_len, selected_y, sentence2_len, logpz, zsum.float(), zdiff.float()
        
        # return selected_x (var), sentence1_len (numpy), selected_y (var), sentence2_len (numpy), selector_loss (var of size 1)
        return selected_x, sentence1_len, selected_y, sentence2_len, logpz, zsum, zdiff

        
        # return , zdiff1, zdiff2
