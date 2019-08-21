###############################################################################
# Author: Wasi Ahmad
# Project: Biattentive Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script contains code to train the model.
###############################################################################

import time, helper, torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable


class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, optimizer, selector, optimizer_selector, dictionary, config, best_acc):
        self.model = model
        self.selector = selector
        self.dictionary = dictionary
        self.config = config
        self.criterion = nn.CrossEntropyLoss(size_average = False)
        if self.config.cuda:
            self.criterion = self.criterion.cuda()
        self.optimizer = optimizer
        self.optimizer_selector = optimizer_selector
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

                self.optimizer_selector.param_groups[0]['lr'] = self.optimizer_selector.param_groups[0]['lr'] * self.config.lr_decay \
                    if epoch > start_epoch and 'sgd' in self.config.optimizer else self.optimizer_selector.param_groups[0]['lr']
              
                if 'sgd' in self.config.optimizer:
                    print('Selector and Model Learning rates are : {0} {0}'.format(self.optimizer_selector.param_groups[0]['lr'], self.optimizer.param_groups[0]['lr']))
                self.train(train_corpus)
                # training epoch completes, now do validation
                if self.config.save_selection==1:
                    print("done all epoch")
                    break # just load a trained selector and save its  selected x , y
                print('\nVALIDATING : Epoch ' + str((epoch + 1)))
                dev_acc = self.validate(dev_corpus)
                self.dev_accuracies.append(dev_acc)
                print('validation acc = %.2f%%' % dev_acc)
                print('\nTESTING : Epoch ' + str((epoch + 1)))
                test_acc = self.validate(test_corpus)
                print('Test acc = %.2f%%' % test_acc)


                # helper.print_trainable_model_params(self.model, how_many=10)


                # save model if dev accuracy goes up
                if self.best_dev_acc < dev_acc:

                    self.best_dev_acc = dev_acc
                    file_path = self.config.output_base_path+self.config.task+'/'+self.config.model_file_name
                    if file_path.endswith('.pth.tar')==False:
                        file_path += 'model_best.pth.tar'

                    helper.save_checkpoint({
                        'epoch': (epoch + 1),
                        'state_dict': self.model.state_dict(),
                        'selector': self.selector.state_dict(),
                        'best_acc': self.best_dev_acc,
                        'optimizer': self.optimizer.state_dict(),
                        'optimizer_selector': self.optimizer_selector.state_dict()
                    }, file_path)

                    print('saved model as: ', file_path)
                    self.times_no_improvement = 0
                else:
                    if 'sgd' in self.config.optimizer:
                        self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] / self.config.lrshrink
                        self.optimizer_selector.param_groups[0]['lr'] = self.optimizer_selector.param_groups[0]['lr'] / self.config.lrshrink
                        print('Shrinking lr by : {0}. New lr = {1} and {1}'.format(self.config.lrshrink, self.optimizer_selector.param_groups[0]['lr'],
                                                                           self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < self.config.minlr:
                            self.stop = True
                    if 'adam' in self.config.optimizer:
                        self.times_no_improvement += 1
                        # early stopping (at 'n'th decrease in accuracy)
                        if self.times_no_improvement == self.config.early_stop:
                            self.stop = True
                # save the train loss and development accuracy plot
                helper.save_plot(self.train_accuracies, self.config.output_base_path+self.config.task+'/', 'training_acc_plot_', epoch + 1)
                helper.save_plot(self.dev_accuracies, self.config.output_base_path+self.config.task+'/', 'dev_acc_plot_', epoch + 1)
            else:
                break

    def train(self, train_corpus):
        if self.config.task=='IMDB': force_min_sen_len = 400
        else: force_min_sen_len = -1

        # Turn on training mode which enables dropout.
        self.selector.train()
        self.model.train()

        # Splitting the data in batches
        shuffle = True
        if self.config.task=='sst' and self.config.save_selection==1: shuffle=False
        print('shuffle', shuffle)
        train_batches = helper.batchify(train_corpus.data, self.config.batch_size, shuffle)
        print('number of train batches = ', len(train_batches))

        # for d in train_batches[0]:
        #     print('tb: ', d.print_())

        start = time.time()
        print_acc_total = 0
        plot_acc_total = 0

        num_batches = len(train_batches)
        num_tokens = 0
        selected_tokens = 0

        selected_x_vars = []
        sentence1_len_list = []
        selected_y_vars = []
        sentence2_len_list = []

        labels_all = []

        # for d in train_batches[1 - 1]:
        #         print('tb: ', d.print_())

        for batch_no in range(1, num_batches + 1):
            # Clearing out all previous gradient computations.
            self.optimizer.zero_grad()
            self.optimizer_selector.zero_grad()
            # for d in train_batches[batch_no - 1]:
            #     print('tb: ', d.print_())
            train_sentences1, sent_len1, train_sentences2, sent_len2, train_labels = helper.batch_to_tensors(
                train_batches[batch_no - 1], self.dictionary, force_min_sen_len=force_min_sen_len)
            # exit()
            if self.config.cuda:
                train_sentences1 = train_sentences1.cuda()
                train_sentences2 = train_sentences2.cuda()
                train_labels = train_labels.cuda()

            assert train_sentences1.size(0) == train_sentences2.size(0)
            selected_x, sentence1_len, selected_y, sentence2_len, logpz, zsum, zdiff = self.selector(train_sentences1, sent_len1, train_sentences2, sent_len2, is_train=1)
            
            if self.config.save_selection:
                sentence1_len_list.append(sentence1_len.tolist())
                selected_x_vars.append(selected_x)
                sentence2_len_list.append(sentence2_len.tolist())
                selected_y_vars.append(selected_y)
                labels_all.append(train_labels)

                continue #no train is needed just load already trained selector and save the selction using it

            score = self.model(selected_x, sentence1_len, selected_y, sentence2_len)
            n_correct = (torch.max(score, 1)[1].view(train_labels.size()).data == train_labels.data).sum()
            
            

            ############################ custom new_loss ############################
            ############################ custom new_loss ############################
            loss_vec = self.criterion(score, train_labels) 
            cost_e = loss_vec.mean()
            # if batch_no==0: 
            #     print ('loss_vec size: ', loss_vec.size(), ' cost_e : ', cost_e)
            cost_e.backward(retain_graph=True)

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
            grad_norm = clip_grad_norm(filter(lambda p: p.requires_grad, self.model.parameters()), self.config.max_norm)
            # print (batch_no, grad_norm)
            self.optimizer.step()

            coherent_factor = self.config.sparsity * self.config.coherent
            cost_vec = loss_vec + zsum * self.config.sparsity + zdiff * coherent_factor
            cost_logpz = (cost_vec * logpz).mean()
        
            cost_g = cost_logpz.mean()
            # if batch_no==0: 
            #     print ('cost_vec size: ', cost_vec.size(), ' cost_g size: ', cost_g)
            cost_g.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
            clip_grad_norm(filter(lambda p: p.requires_grad, self.selector.parameters()), self.config.max_norm)
            self.optimizer_selector.step()


            ############################ custom new_loss ############################
            ############################ custom new_loss ############################
            


            print_acc_total += 100. * n_correct / len(train_batches[batch_no - 1])
            plot_acc_total += 100. * n_correct / len(train_batches[batch_no - 1])
            num_tokens += sum(sent_len1)+sum(sent_len2)
            selected_tokens+= sum(sentence1_len)+sum(sentence2_len)

            if batch_no % self.config.print_every == 0 or self.config.debug: 
                print_acc_avg = print_acc_total / self.config.print_every
                print_acc_total = 0
                p = 100.0 * selected_tokens/num_tokens
                print('%s (%d %d%%)  (%.2f%%) %.2f' % (
                    helper.show_progress(start, batch_no / num_batches), batch_no,
                    batch_no / num_batches * 100, p, print_acc_avg))

            if batch_no % self.config.plot_every == 0:
                plot_acc_avg = plot_acc_total / self.config.plot_every
                self.train_accuracies.append(plot_acc_avg)
                plot_acc_total = 0

        if self.config.save_selection==1:
            helper.save_selected_x_y_sent1_len_sent2_len(selected_x_vars, selected_y_vars, sentence1_len_list, sentence2_len_list,  labels_all, self.config, sparsity = self.config.sparsity, coherent = self.config.coherent, dictionary=self.dictionary)
            print("done an epoch")

    def validate(self, dev_corpus):
        if self.config.task=='IMDB': force_min_sen_len = 400
        else: force_min_sen_len = -1

        # Turn on evaluation mode which disables dropout.
        self.selector.eval()
        self.model.eval()
        start = time.time()
        selection_time = 0
        selected_tokens = num_tokens = print_acc_total = 0

        dev_batches = helper.batchify(dev_corpus.data, self.config.batch_size)
        print('number of dev batches = ', len(dev_batches))

        num_batches = len(dev_batches)
        n_correct, n_total = 0, 0
        for batch_no in range(1, num_batches + 1):
            dev_sentences1, sent_len1, dev_sentences2, sent_len2, dev_labels = helper.batch_to_tensors(
                dev_batches[batch_no - 1], self.dictionary, True, force_min_sen_len=force_min_sen_len)
            if self.config.cuda:
                dev_sentences1 = dev_sentences1.cuda()
                dev_sentences2 = dev_sentences2.cuda()
                dev_labels = dev_labels.cuda()

            assert dev_sentences1.size(0) == dev_sentences2.size(0)
            strat_t = time.time()
            selected_x, sentence1_len, selected_y, sentence2_len, logpz, zsum, zdiff = self.selector(dev_sentences1, sent_len1, dev_sentences2, sent_len2)
            selection_time+= time.time()-strat_t
            score = self.model(selected_x, sentence1_len, selected_y, sentence2_len)
            n_correct += (torch.max(score, 1)[1].view(dev_labels.size()).data == dev_labels.data).sum()
            n_total += len(dev_batches[batch_no - 1])

            print_acc = 100. * n_correct / n_total
            num_tokens += sum(sent_len1)+sum(sent_len2)
            selected_tokens+= sum(sentence1_len)+sum(sentence2_len)

            if batch_no % self.config.print_every == 0 or self.config.debug:
                p = 100.0 * selected_tokens/num_tokens
                print('%s (%d %d%%) (%.2f) %.2f' % (
                    helper.show_progress(start, batch_no / num_batches), batch_no,
                    batch_no / num_batches * 100, p, print_acc))

        p = 100.0 * selected_tokens/num_tokens
        print('selection_time: %s Total:  %s (%d %d%%) (%.2f) %.2f' % (
                    helper.convert_to_minutes(selection_time),
                    helper.show_progress(start, batch_no / num_batches), batch_no,
                    batch_no / num_batches * 100, p, print_acc))




        return 100. * n_correct / n_total
