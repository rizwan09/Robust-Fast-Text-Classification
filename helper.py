###############################################################################
# Author: Wasi Ahmad
# Project: Biattentive Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script provides general purpose utility functions that
# may come in handy at any point in the experiments.
###############################################################################

import re, os, json, glob, pickle, inspect, math, time, torch, util
import numpy as np
from torch import optim
from nltk import word_tokenize
from collections import OrderedDict
from torch.autograd import Variable
import matplotlib as mpl
import torch.nn.functional as f
from scipy.sparse import csr_matrix, lil_matrix


mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

args = util.get_args()
np.random.seed(args.seed)

def load_word_embeddings(directory, file, dictionary):
    embeddings_index = {}
    f = open(os.path.join(directory, file))
    for line in f:
        word, vec = line.split(' ', 1)
        if word in dictionary:
            embeddings_index[word] = np.array(list(map(float, vec.split())))
    f.close()
    return embeddings_index


def save_word_embeddings(directory, file, embeddings_index):
    f = open(os.path.join(directory, file), 'w')
    for word, vec in embeddings_index.items():
        f.write(word + ' ' + ' '.join(str(x) for x in vec) + '\n')
    f.close()


def load_checkpoint(filename, from_gpu=True):
    """Load a previously saved checkpoint."""
    assert os.path.exists(filename)
    if from_gpu:
        return torch.load(filename)
    else:
        return torch.load(filename, map_location=lambda storage, loc: storage)


def save_checkpoint(state, filename='./checkpoint.pth.tar'):
    if os.path.isfile(filename):
        os.remove(filename)
    torch.save(state, filename)


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = list(inspect.signature(optim_fn.__init__).parameters.keys())
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


def load_model_states_from_checkpoint(model, filename, tag, from_gpu=True):
    """Load model states from a previously saved checkpoint."""
    assert os.path.exists(filename)
    if from_gpu:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint[tag])

def load_model(model, filename, tag, from_gpu=True):
    print ('loading: ', filename)
    assert os.path.exists(filename)
    if from_gpu:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint[tag])


def load_selector_classifier_states_from_checkpoint(selector, selector_filename, tag_selector, model, filename, tag, from_gpu=True):
    """Load model states from a previously saved checkpoint."""
    assert os.path.exists(selector_filename)
    assert os.path.exists(filename)
    if from_gpu:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    selector.load_state_dict(checkpoint[tag_selector])
    model.load_state_dict(checkpoint[tag])



def load_model_states_without_dataparallel(model, filename, tag):
    """Load a previously saved model states."""
    assert os.path.exists(filename)
    checkpoint = torch.load(filename)
    new_state_dict = OrderedDict()
    for k, v in checkpoint[tag].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


def save_object(obj, filename):
    """Save an object into file."""
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def load_object(filename):
    """Load object from file."""
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj


def count_parameters(model):
    param_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_dict[name] = np.prod(param.size())
    return param_dict


def print_trainable_model_params(model, how_many=-1):
    c=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # if how_many and param.size()[0]>how_many: 
            print(name,param.size(), param)
            # else:
            #     print(name,param.size(), param)
            c+=1
            if how_many>0 and c>how_many: break


def tokenize(s, tokenize):
    """Tokenize string."""
    if tokenize:
        return word_tokenize(s)
    else:
        return s.split()


def initialize_out_of_vocab_words(dimension, choice='zero'):
    """Returns a vector of size dimension given a specific choice."""
    if choice == 'random':
        """Returns a random vector of size dimension where mean is 0 and standard deviation is 1."""
        return np.random.normal(size=dimension)
    elif choice == 'zero':
        """Returns a vector of zeros of size dimension."""
        return np.zeros(shape=dimension)


def batchify(data, bsz, shuffle=True):
    """Transform data into batches."""
    if shuffle: np.random.shuffle(data)
    batched_data = []
    for i in range(len(data)):
        # print(' ddata[i]: ', data[i].print_())
        if i % bsz == 0:
            batched_data.append([data[i]])
        else:
            batched_data[len(batched_data) - 1].append(data[i])
    if len(batched_data[-1])==1: return batched_data[:-1]
    return batched_data


def print_corpus(data):
    """Transform data into batches."""
    batched_data = []
    for i in range(len(data)):
        print(' ddata[i]: ', data[i].print_())

def convert_to_minutes(s):
    """Converts seconds to minutes and seconds"""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def show_progress(since, percent):
    """Prints time elapsed and estimated time remaining given the current time and progress in %"""
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (convert_to_minutes(s), convert_to_minutes(rs))


def save_plot(points, filepath, filetag, epoch):
    """Generate and save the plot"""
    path_prefix = os.path.join(filepath, filetag)
    path = path_prefix + 'epoch_{}.png'.format(epoch)
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.plot(points)
    fig.savefig(path)
    plt.close(fig)  # close the figure
    for f in glob.glob(path_prefix + '*'):
        if f != path:
            os.remove(f)


def show_plot(points):
    """Generates plots"""
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def sequence_to_tensor(sequence, max_sent_length, dictionary):
    """Convert a sequence of words to a tensor of word indices."""
    sen_rep = torch.LongTensor(max_sent_length).zero_()
    for j in range(len(sequence)):
        if dictionary.contains(sequence[j]):
            sen_rep[j] = dictionary.word2idx[sequence[j]]
    return sen_rep


def get_L1_selected(sequence, sen_rep, dictionary, non_zero_indices=[], row = None):
    assert  row!=None
    assert non_zero_indices!= None

    for i in range(len(sequence)):
        if dictionary.contains(sequence[i]):
            if dictionary.word2idx[sequence[i]] in non_zero_indices:
                sen_rep[row, dictionary.word2idx[sequence[i]]] = dictionary.word2idx[sequence[i]]
            
    return sen_rep
     

def sequence_to_one_hot_encoded(sequence, sen_rep, dictionary, non_zero_indices=[], row = None):
    """Convert a sequence of words to a tensor of word indices."""
    assert  row!=None
    # import pdb
    # pdb.set_trace()

    for i in range(len(sequence)):
        if dictionary.contains(sequence[i]):
            if len(non_zero_indices)>0:
                if dictionary.word2idx[sequence[i]] in non_zero_indices:
                    sen_rep[row, dictionary.word2idx[sequence[i]]] = 1
            else:
                sen_rep[row, dictionary.word2idx[sequence[i]]] = 1

    return sen_rep

def get_max_length(batch):
    max_sent_length1, max_sent_length2 = 0, 0
    for item in batch:
        # print(' in get max len item:', item.sentence1)
        if max_sent_length1 < len(item.sentence1):
            max_sent_length1 = len(item.sentence1)
        if max_sent_length2 < len(item.sentence2):
            max_sent_length2 = len(item.sentence2)

    return max_sent_length1, max_sent_length2


def selected_text_l1(batch, dictionary, non_zero_indices=None, iseval=False, force_min_sen_len=-1):
    """Convert a list of sequences to a list of tensors."""

    # for d in batch:
    #     print('b: ', d.print_())

    max_sent_length1 = len(dictionary.word2idx)
    
    all_sentences1 = lil_matrix((len(batch), max_sent_length1))
    labels = np.zeros(len(batch), dtype=int)
    for i in range(len(batch)):
        get_L1_selected(batch[i].sentence1, all_sentences1, dictionary, non_zero_indices, i)
        labels[i] = batch[i].label
    return all_sentences1, labels


def get_L1_selected_faster(sequence, sen_rep, dictionary, selected_flag=[], row = None):
    assert  row!=None
    assert non_zero_indices!= None
    

    for i in range(len(sequence)):
        if dictionary.contains(sequence[i]):
            if selected_flag[ dictionary.word2idx[sequence[i]] ]:
                sen_rep[row, dictionary.word2idx[sequence[i]]] = dictionary.word2idx[sequence[i]]
            
    return sen_rep

def selected_text_l1_faster(batch, dictionary, non_zero_indices=None, iseval=False, force_min_sen_len=-1):
    """Convert a list of sequences to a list of tensors."""

    selected_flag = np.zeros(len(dictionary.word2idx), dtype=int)
    for idx in non_zero_indices:
        selected_flag[idx] = 1
    max_sent_length1 = len(dictionary.word2idx)
    
    all_sentences1 = lil_matrix((len(batch), max_sent_length1))
    labels = np.zeros(len(batch), dtype=int)
    for i in range(len(batch)):
        get_L1_selected_faster(batch[i].sentence1, all_sentences1, dictionary, selected_flag, i)
        labels[i] = batch[i].label
        print()
    return all_sentences1, labels

def batch_to_one_hot_encoded(batch, dictionary, non_zero_indices=None, iseval=False, force_min_sen_len=-1):
    """Convert a list of sequences to a list of tensors."""

    # for d in batch:
    #     print('b: ', d.print_())

    max_sent_length1 = len(dictionary.word2idx)
    
    all_sentences1 = lil_matrix((len(batch), max_sent_length1))
    labels = np.zeros(len(batch), dtype=int)
    for i in range(len(batch)):
        sequence_to_one_hot_encoded(batch[i].sentence1, all_sentences1, dictionary, non_zero_indices, i)
        labels[i] = batch[i].label
    return all_sentences1, labels
    


def batch_to_tensors(batch, dictionary, iseval=False, force_min_sen_len=-1):
    """Convert a list of sequences to a list of tensors."""

    # for d in batch:
    #     print('b: ', d.print_())

    # import pdb
    # pdb.set_trace()

    max_sent_length1, max_sent_length2 = get_max_length(batch)
    if force_min_sen_len>max_sent_length1: max_sent_length1 = force_min_sen_len
    if force_min_sen_len>max_sent_length2: max_sent_length2 = force_min_sen_len

    all_sentences1 = torch.LongTensor(len(batch), max_sent_length1)
    sent_len1 = np.zeros(len(batch), dtype=np.int)
    all_sentences2 = torch.LongTensor(len(batch), max_sent_length2)
    sent_len2 = np.zeros(len(batch), dtype=np.int)
    labels = torch.LongTensor(len(batch))
    for i in range(len(batch)):

        sent_len1[i], sent_len2[i] = len(batch[i].sentence1), len(batch[i].sentence2)


        if force_min_sen_len>0 and (force_min_sen_len>sent_len1[i]): sent_len1[i] = force_min_sen_len
        if force_min_sen_len>0 and (force_min_sen_len>sent_len2[i]): sent_len2[i] = force_min_sen_len

        trim_flag = False
        if force_min_sen_len>0 and (sent_len1[i]>max_sent_length1 or sent_len2[i]>max_sent_length1):
            sent_len1[i] = force_min_sen_len
            sent_len2[i] = force_min_sen_len
            trim_flag = True

        if not trim_flag:
            all_sentences1[i] = sequence_to_tensor(batch[i].sentence1, max_sent_length1, dictionary)
            all_sentences2[i] = sequence_to_tensor(batch[i].sentence2, max_sent_length2, dictionary)
        else:
            all_sentences1[i] = sequence_to_tensor(batch[i].sentence1[:max_sent_length1], max_sent_length1, dictionary)
            all_sentences2[i] = sequence_to_tensor(batch[i].sentence2[:max_sent_length1], max_sent_length2, dictionary)

        labels[i] = batch[i].label

    if iseval:
        return Variable(all_sentences1, volatile=True), sent_len1, Variable(all_sentences2, volatile=True), sent_len2, \
               Variable(labels, volatile=True)
    else:
        return Variable(all_sentences1), sent_len1, Variable(all_sentences2), sent_len2, Variable(labels)



def get_selected_variable(embedded_x, selection_x, cuda):
    # embedded_x (batch x sentence len x emsize)
    # selection_x (batch x sentence len)
    # output:  (batch x max sentence len in selection_x x emsize)
    r = torch.FloatTensor(embedded_x.size(0), int(selection_x.sum(0).max()), embedded_x.size(2)).zero_()
    
    for i in range(embedded_x.size(0)):
        for j in range(embedded_x.size(1)):
            if int(selection_x_list[i][j])==1: r[i][j]= embedded_x_list[i][j].data
    rt = Variable(torch.Tensor(r))
    if cuda: rt = rt.cuda()
    return rt


def get_splited_imdb_data(file_name, data_name='IMDB', SAG=False):
    if data_name=='IMDB':
        train_d =[]
        dev_d =[]
        test_d = []
        print(file_name)
        with open(file_name, 'rb') as f:
            
            
            if file_name.endswith('.p'): 
                x = pickle.load(f, encoding="latin1")
                all_d = x[0]
            else: 
                with open(file_name) as json_file:  
                    all_d = json.load(json_file)

            for line in all_d:
                if line['split']==0: 
                    train_d.append(line)
                    if SAG:
                        for i in range(args.nversion):
                            temp_line = line
                            x = temp_line['text']
                            new_x=''

                            # a = 0
                            # for b in range(len(x)):
                            #     w = x[b]
                            #     if w == ',':
                            #         rnd = np.random.rand()
                            #         # if(i==3 and j==3):print i, j, rnd
                            #         if (rnd>args.blankout_prob ): # the reason is at least one sentence should be included
                            #             for ww in x[a:b+1]: new_x+=ww
                            #         a = b+1 

                            for chunk in x.split(','):
                                rnd = np.random.rand()
                                if (rnd>args.blankout_prob ):
                                    new_x+=chunk

                            if new_x=='': new_x=x
                                

                            if len(new_x) > 0:
                                temp_line['text'] = new_x
                            train_d.append(temp_line)
                    # if args.SAG:
                    #     print(train_d)
                    #     exit()
                        
                elif line['split']==1: dev_d.append(line)
                else: test_d.append(line)
        return train_d, dev_d, test_d



def get_selected_tensor(result_x, pbx, sentence1, sentence1_len_old, cuda):

    # print (result_x.size())
    # if result_x.size()[0]==1:

    sentences = []
    sent_len =  np.zeros(result_x.size()[0], dtype=np.int)
    for i,s in enumerate(result_x):
        s = s[:sentence1_len_old[i]] #discard padded portion first
        sn = s[(s!=0).detach()] #non_zero elements
        #make sure that atleast one word is selected
        while sn.dim()==0:
            pb = pbx[i,:sentence1_len_old[i]]
            s = sentence1[i,:sentence1_len_old[i]].mul(pb.bernoulli().long())
            sn = s[(s!=0).detach()] #non_zero elements
             
        sentences.append(sn)
        sent_len[i] = sn.size()[0]
    max_sent_length = max(sent_len)
    sentences_tensor = torch.LongTensor(result_x.size()[0], int(max_sent_length)).zero_()
    for i in range(result_x.size()[0]):
        sentences_tensor[i,:sent_len[i]] = sentences[i].data
    sent_var = Variable(sentences_tensor)
    if cuda: sent_var = sent_var.cuda()
    return sent_var, sent_len


def binary_cross_entropy(pbx, targets, size_average = True, reduce = False):
    if reduce==True:
        return f.binary_cross_entropy(pbx, targets, size_average = size_average)
    else:
        return  -(targets*torch.log(pbx)+ (1-targets)*torch.log(1-pbx))


def pad_variables(variable_lists, cuda=True, batch_first=True, padding_value=0):
    # variable_lists [ v1(batch x len), v2(batch x len), ..] all are long Variables
    batch_size =0
    max_len = 0
    for v in variable_lists:

        batch_size+=v.size()[0]
        if v.size()[1]>max_len: max_len=v.size()[1]
    nv = Variable(torch.LongTensor(batch_size, max_len).zero_())
    if cuda: nv =nv.cuda()
    # print('initial size :', nv.size())
    i = 0
    for v in variable_lists:
        # print('step des size :', nv[i:i+v.size()[0], :v.size()[1]].size(), ' source size ', v.size(), ' i: ', i, ' to ', i+v.size()[0])
        nv[i:i+v.size()[0], :v.size()[1]] = v.data
        i+=v.size()[0]
    return nv

def save_selected_x_y_sent1_len_sent2_len(selected_x_list, selected_y_list, sent1_len_list, sent2_len_list,  labels_all, args, sparsity = -1.0, coherent = 2.0, file_prefix = 'train_selected_', dictionary=None, file_suffix='.txt'):
    # selected_x_list (batch x batch_size x sent len)
    if not dictionary: 
        dir_path = args.output_base_path+args.task+'/'+'selected/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_name = dir_path+file_prefix+'sparsity_'+str(sparsity)+'_coherent_'+str(coherent)+file_suffix
        torch.save((selected_x_list,selected_y_list, sent1_len_list, sent2_len_list), file_name)
        print('saved as ', file_name)
    else:
        dir_path = args.output_base_path+args.task+'/'+'selected/'
        file_name = dir_path+file_prefix+'sparsity_'+str(sparsity)+'_coherent_'+str(coherent)+file_suffix
        objs = []
        for batch_no in range(len(selected_x_list)):
            for row,instance in enumerate(selected_x_list[batch_no]):
                s=""
                for idx in instance.cpu().data.numpy():
                    if idx==0:break
                    if '<s>'not in dictionary.idx2word[idx] and  '</s>' not in dictionary.idx2word[idx]: s += dictionary.idx2word[idx]+' '
                # print(' batch no: ', batch_no, ' row: ', row, 'text: ', s, " y: ", str(int(labels_all[batch_no][row].cpu().data)))
                objs.append({"text": s, "y":str(int(labels_all[batch_no][row].cpu().data)), "split":0})
            # exit()
        with open(file_name, 'w') as f:
            json.dump(objs, f)
            print('saved as ', file_name)

def build_new_dict(dict1, dict2, non_zero_indices):
    for idx in non_zero_indices:
        dict1.idx2word.append(dict2.dx2word[idx])
        dict1.word2idx[dict2.dx2word[idx]] = len(dict1.idx2word) - 1
   

def load_selected_x_y_sent1_len_sent2_len( args, sparsity = -1.0, coherent = 2.0, file_prefix = 'train_selected_', file_suffix = '.pth.tar'):
    dir_path = args.output_base_path+args.task+'/'+'selected/'
    file_name = dir_path+file_prefix+'sparsity_'+str(sparsity)+'_coherent_'+str(coherent)+file_suffix
    assert os.path.exists(dir_path)
    file_name = dir_path+file_prefix+'sparsity_'+str(sparsity)+'_coherent_'+str(coherent)+file_suffix
    selected_x_list,selected_y_list, sent1_len_list, sent2_len_list,= torch.load(file_name)
    print('loaded ', file_name)
    return selected_x_list,selected_y_list, sent1_len_list, sent2_len_list,

def load_all_selected_x_y_sent1_len_sent2_len(sparsity_all, coherent_all, args, file_prefix = 'train_selected_', file_suffix = '.pth.tar', sparsity_all2 = [0.00075], coherent_all2=[1.0]):
    all_selected_x_list =[]
    all_selected_y_list =[]
    all_sent1_len_list = []
    all_sent2_len_list = []
    assert len(coherent_all) ==1
    for sp in (sparsity_all):
        selected_x_list, selected_y_list, sent1_len_list, sent2_len_list = load_selected_x_y_sent1_len_sent2_len(args, sparsity = sp, coherent = 2.0)
        all_selected_x_list.append(selected_x_list)
        all_selected_y_list.append(selected_y_list)
        all_sent1_len_list.append(sent1_len_list)
        all_sent2_len_list.append(sent2_len_list)
    for sp in (sparsity_all2):
        selected_x_list, selected_y_list, sent1_len_list, sent2_len_list = load_selected_x_y_sent1_len_sent2_len(args, sparsity = sp, coherent = 1.0)
        all_selected_x_list.append(selected_x_list)
        all_selected_y_list.append(selected_y_list)
        all_sent1_len_list.append(sent1_len_list)
        all_sent2_len_list.append(sent2_len_list)
    assert len(all_selected_x_list) ==  len(sparsity_all)*len(coherent_all)+len(sparsity_all2)*len(coherent_all2)
    assert len(all_selected_y_list) ==  len(sparsity_all)*len(coherent_all)+len(sparsity_all2)*len(coherent_all2)
    return all_selected_x_list, all_selected_y_list, all_sent1_len_list, all_sent2_len_list# len(sparsity_all)*len(coherent_all) x batch x var(len)

        





