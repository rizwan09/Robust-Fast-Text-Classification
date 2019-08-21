###############################################################################
# Author: Wasi Ahmad
# Project: Biattentive Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script is the entry point of the entire pipeline.
###############################################################################


import util, helper, data, os, sys, numpy, torch, pickle, json
from torch import optim
from torch.autograd import Variable
import torch, helper
import torch.nn as nn
import torch.nn.functional as f
from collections import OrderedDict
from nn_layer import EmbeddingLayer, WE_Selector
import time
from torch.nn.utils import clip_grad_norm

args = util.get_args()
# if output directory doesn't exist, create it
if not os.path.exists(args.output_base_path):
    os.makedirs(args.output_base_path)

# Set the random seed manually for reproducibility.
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

print('\ncommand-line params : {0}\n'.format(sys.argv[1:]))
print('{0}\n'.format(args))

###############################################################################
# Load data
###############################################################################

# load train and dev dataset
train_corpus = data.Corpus(args.tokenize)
dev_corpus = data.Corpus(args.tokenize)
test_corpus = data.Corpus(args.tokenize)

train_corpus.parse(args.output_base_path + args.task + '/dev_train.txt', args.task, args.max_example)
dev_corpus.parse(args.output_base_path + args.task + '/word_dev.txt', args.task, args.tokenize)
test_corpus.parse(args.output_base_path + args.task + '/word_test.txt', args.task, args.tokenize)

# helper.print_corpus(train_corpus.data)


print('train set size = ', len(train_corpus.data),)
print('development set size = ', len(dev_corpus.data))
print('test set size = ', len(test_corpus.data))


if os.path.exists(args.output_base_path + args.task+'/'+ 'word_dictionary.p'):
    print('loading dictionary')
    dictionary = helper.load_object(args.output_base_path + args.task+'/'+ 'word_dictionary.p') 
else:
    dictionary = data.Dictionary()
    dictionary.build_dict(train_corpus.data + dev_corpus.data + test_corpus.data, args.max_words)
    helper.save_object(dictionary, args.output_base_path + args.task+'/' + 'word_dictionary.p')


    
print('vocabulary size = ', len(dictionary))

embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file, dictionary.word2idx)
print('number of OOV words = ', len(dictionary) - len(embeddings_index))

# ###############################################################################
# # Build the model
# ###############################################################################



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
        self.emsize = args.emsize
        self.num_labels = args.num_class
        self.linear= nn.Linear(self.emsize, self.num_labels)
    


    def forward(self, sentence1, threshold = 0.5, is_train = 0):
        embedded_x1 = self.embedding(sentence1)
        score  = self.linear(embedded_x1)
        # print("linear size: ", score.size())
        score= score.squeeze(1)
        # print('output size: ', score.size())

        return score

        
        # return , zdiff1, zdiff2




selector = Selector(dictionary, embeddings_index, args)
print (selector)
optim_fn_selector, optim_params_selector = helper.get_optimizer(args.optimizer)
optimizer_selector = optim_fn_selector(filter(lambda p: p.requires_grad, selector.parameters()), **optim_params_selector)

best_acc = 0
param_dict_selector = helper.count_parameters(selector)
print('number of trainable parameters = ',  numpy.sum(list(param_dict_selector.values())) )

if args.cuda:
    torch.cuda.set_device(args.gpu)
    selector = selector.cuda()
    

# ###############################################################################
# # Train the model
# ###############################################################################
###############################################################################
# Author: Wasi Ahmad
# Project: Biattentive Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script contains code to train the model.
###############################################################################


class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, optimizer, dictionary, config, best_acc):
        self.model = model
        self.dictionary = dictionary
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        if self.config.cuda:
            self.criterion = self.criterion.cuda()

        self.optimizer = optimizer
        self.best_dev_acc = best_acc
        self.times_no_improvement = 0
        self.stop = False
        self.train_accuracies = []
        self.dev_accuracies = []

    def train_epochs(self, train_corpus, dev_corpus, test_corpus, start_epoch, n_epochs):
        """Trains model for n_epochs epochs"""
        for epoch in range(start_epoch, start_epoch + n_epochs):
            if not self.stop:
                print('\nTRAINING : Epoch ' + str((epoch + 1)))
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * self.config.lr_decay \
                    if epoch > start_epoch and 'sgd' in self.config.optimizer else self.optimizer.param_groups[0]['lr']
                if 'sgd' in self.config.optimizer:
                    print('Learning rate : {0}'.format(self.optimizer.param_groups[0]['lr']))
                try:
                    self.train(train_corpus, epoch+1)
                except KeyboardInterrupt:
                    print('-' * 89)
                    print('Exiting from training early')
                # training epoch completes, now do validation
                print('\nVALIDATING : Epoch ' + str((epoch + 1)))
                dev_acc = -1
                try:
                    dev_acc = self.validate(dev_corpus)
                    self.dev_accuracies.append(dev_acc)
                    print('validation acc = %.2f%%' % dev_acc)
                except KeyboardInterrupt:
                    print('-' * 89)
                    print('Exiting from dev early')
                
            
                try:
                    test_acc = self.validate(test_corpus)
                    print('validation acc = %.2f%%' % test_acc)
                except KeyboardInterrupt:
                    print('-' * 89)
                    print('Exiting from testing early')
                
    
                # save model if dev accuracy goes up
                if self.best_dev_acc < dev_acc and dev_acc!=-1:
                    self.best_dev_acc = dev_acc
                    file_path = self.config.output_base_path+self.config.task+'/'+self.config.model_file_name
                    if file_path.endswith('.pth.tar')==False:
                        file_path += 'model_best.pth.tar'

                    helper.save_checkpoint({
                        'epoch': (epoch + 1),
                        'state_dict': self.model.state_dict(),
                        'best_acc': self.best_dev_acc,
                        'optimizer': self.optimizer.state_dict() 
                    }, file_path)
                    print('model saved as: ', file_path)
                    self.times_no_improvement = 0
                else:
                    if 'sgd' in self.config.optimizer:
                        self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] / self.config.lrshrink
                        print('Shrinking lr by : {0}. New lr = {1}'.format(self.config.lrshrink,
                                                                           self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < self.config.minlr:
                            self.stop = True
                    if 'adam' in self.config.optimizer:
                        self.times_no_improvement += 1
                        # early stopping (at 'n'th decrease in accuracy)
                        if self.times_no_improvement == self.config.early_stop:
                            self.stop = True
                # save the train loss and development accuracy plot
                helper.save_plot(self.train_accuracies, self.config.output_base_path, 'training_acc_plot_', epoch + 1)
                helper.save_plot(self.dev_accuracies, self.config.output_base_path, 'dev_acc_plot_', epoch + 1)
            else:
                break

    def train(self, train_corpus, epoch):
        # Turn on training mode which enables dropout.
        self.model.train()

        # Splitting the data in batches
        shuffle = True
        # if self.config.task == 'sst': shuffle = False
        print(shuffle)

        train_batches = helper.batchify(train_corpus.data, self.config.batch_size, shuffle)
        print('number of train batches = ', len(train_batches))

        start = time.time()
        print_acc_total = 0
        plot_acc_total = 0

        num_batches = len(train_batches)
        for batch_no in range(1, num_batches + 1):
            # Clearing out all previous gradient computations.
            self.optimizer.zero_grad()
            train_sentences1, sent_len1, train_sentences2, sent_len2, train_labels = helper.batch_to_tensors(
                train_batches[batch_no - 1], self.dictionary)
            if self.config.cuda:
                train_sentences1 = train_sentences1.cuda()
                train_sentences2 = train_sentences2.cuda()
                train_labels = train_labels.cuda()

            assert train_sentences1.size(0) == train_sentences2.size(0)

            # print(' train label size: ', train_labels.size(), ' train data size: ', train_sentences1.size())
            # print(' labels: ', train_labels)
            score = self.model(train_sentences1)
            n_correct = (torch.max(score, 1)[1].view(train_labels.size()).data == train_labels.data).sum()
            # print (' score size ', score.size(), train_labels.size())
            loss = self.criterion(score, train_labels)


            if loss.size(0) > 1:
                loss = loss.mean()
            # print ('loss:', loss)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
            grad_norm = clip_grad_norm(filter(lambda p: p.requires_grad, self.model.parameters()), self.config.max_norm)
            # if epoch==11:
            # print(batch_no, grad_norm)
            self.optimizer.step()

            print_acc_total += 100. * n_correct / len(train_batches[batch_no - 1])
            plot_acc_total += 100. * n_correct / len(train_batches[batch_no - 1])

            if batch_no % self.config.print_every == 0:
                print_acc_avg = print_acc_total / self.config.print_every
                print_acc_total = 0
                print('%s (%d %d%%) %.2f' % (
                    helper.show_progress(start, batch_no / num_batches), batch_no,
                    batch_no / num_batches * 100, print_acc_avg))

            if batch_no % self.config.plot_every == 0:
                plot_acc_avg = plot_acc_total / self.config.plot_every
                self.train_accuracies.append(plot_acc_avg)
                plot_acc_total = 0

    def validate(self, dev_corpus):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        dev_batches = helper.batchify(dev_corpus.data, self.config.batch_size)
        print('number of dev batches = ', len(dev_batches))

        num_batches = len(dev_batches)
        n_correct, n_total = 0, 0
        for batch_no in range(1, num_batches + 1):
            dev_sentences1, sent_len1, dev_sentences2, sent_len2, dev_labels = helper.batch_to_tensors(
                dev_batches[batch_no - 1], self.dictionary, True)
            if self.config.cuda:
                dev_sentences1 = dev_sentences1.cuda()
                dev_sentences2 = dev_sentences2.cuda()
                dev_labels = dev_labels.cuda()

            assert dev_sentences1.size(0) == dev_sentences2.size(0)

            score = self.model(dev_sentences1)
            n_correct += (torch.max(score, 1)[1].view(dev_labels.size()).data == dev_labels.data).sum()
            n_total += len(dev_batches[batch_no - 1])

        return 100. * n_correct / n_total



# train = Train(selector, optimizer_selector, dictionary, args, best_acc)
# train.train_epochs(train_corpus, dev_corpus, test_corpus, args.start_epoch, args.epochs)



#### testing 
file_path = args.output_base_path+args.task+'/'+args.model_file_name                 
print('loading selector from: ', file_path)
helper.load_model(selector, file_path, 'state_dict', args.cuda)


selector.eval()

dev_batches = helper.batchify(test_corpus.data, args.batch_size)
print('number of dev batches = ', len(dev_batches))

num_batches = len(dev_batches)
n_correct, n_total = 0, 0
with open('../bcn_output/sst/predicted_text_words_dummy.txt', 'w')as wf:
    for batch_no in range(1, num_batches + 1):
        dev_sentences1, sent_len1, dev_sentences2, sent_len2, dev_labels = helper.batch_to_tensors(
            dev_batches[batch_no - 1], dictionary, True)
        if args.cuda:
            dev_sentences1 = dev_sentences1.cuda()
            dev_sentences2 = dev_sentences2.cuda()
            dev_labels = dev_labels.cuda()

        assert dev_sentences1.size(0) == dev_sentences2.size(0)

        score = selector(dev_sentences1)
        n_correct += (torch.max(score, 1)[1].view(dev_labels.size()).data == dev_labels.data).sum()
        n_total += len(dev_batches[batch_no - 1])

        for (sent1, sel, tl) in zip(dev_sentences1, torch.max(score, 1)[1].view(dev_labels.size()).data, dev_labels.data):
            print(dictionary.idx2word[sent1.data[0]], sel, tl)
            if sel==1:
                wf.write(dictionary.idx2word[sent1.data[0]].lower()+"\n")

print('test final acc: ', 100. * n_correct / n_total)

