###############################################################################
# Author: Wasi Ahmad
# Project: Biattentive Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script is the entry point of the entire pipeline.
###############################################################################


import util, helper, data, os, sys, numpy, torch, pickle, json
import imdb_train as train
from torch import optim
import pdb, pickle, time
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import density
from scipy.sparse import csr_matrix, lil_matrix



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
def eval_routine(corpus, dictionary, model, non_zero_indices=None):
    print('one hot encoding...')
    eval_sentences, eval_labels = helper.batch_to_one_hot_encoded(corpus.data, dictionary, non_zero_indices = non_zero_indices)
    if not non_zero_indices:
        print('Creating CSR sparsing...')
        eval_sentences = csr_matrix(eval_sentences)
    print('Testing...')
    acc = model.score(eval_sentences, eval_labels)
    print(' Accurcay: ', acc)
    return acc

def eval(corpus, dictionary, loaded_model, non_zero_indices):
    start = time.time()
    acc = eval_routine(corpus, dictionary, loaded_model, non_zero_indices=non_zero_indices)
    t = time.time() - start
    print('total test time:  %s ' % helper.convert_to_minutes(t))
    return t, acc

def get_trained_model(c, corpus, dictionary, non_zero_indices):
    model = LogisticRegression(penalty='l1',  tol=0.0001, C=c, fit_intercept=True, \
        intercept_scaling=1, solver='liblinear', max_iter=args.epochs, multi_class='ovr', verbose=1)
    print('one hot encoding...')
    train_sentences1, train_labels = helper.batch_to_one_hot_encoded(corpus.data, dictionary, non_zero_indices = non_zero_indices)
    print('Training...')
    model.fit(train_sentences1, train_labels)
    return 

def save_data_routine( corpus, dictionary, non_zero_indices, file = 'train'):
    selected_flag = numpy.zeros(len(dictionary.word2idx), dtype=int)
    for idx in non_zero_indices:
        selected_flag[idx] = 1
    dir_path = args.output_base_path+args.task
    if '_L1' not in dir_path: dir_path+='_L1'
    file_name = dir_path+"/"+file+'-l1.txt'
    objs = []
    time_needed = time.time()
    for row,instance in enumerate(corpus.data):
        s=""
        for wd in instance.sentence1:
            if wd in dictionary.word2idx:
                if '<s>'not in wd and  '</s>' not in wd and selected_flag[dictionary.word2idx[wd]]: s += wd+' '
        # print('done: ', row, '/', len(train_sentences1))
        # print(' batch no: ', batch_no, ' row: ', row, 'text: ', s, " y: ", str(int(labels_all[batch_no][row].cpu().data)))
        objs.append({"text": s, "y":str(int(instance.label)), "split":0})
        # exit()
    time_needed = time.time() - time_needed
    print("  time neede: ", time_needed)
    with open(file_name, 'w') as wf:
        print(' start writing ')
        for line in objs:
            wf.write(str(int(line['y'])+1)+"\t"+line['text']+"\n")
        print('saved as ', file_name)


    
    
###############################################################################
# Load data
###############################################################################

# load train and dev dataset
train_corpus = data.Corpus(args.tokenize)
dev_corpus = data.Corpus(args.tokenize)
test_corpus = data.Corpus(args.tokenize)

task_names = ['snli', 'multinli'] if args.task == 'allnli' else [args.task]
for task in task_names:
    if 'IMDB' in task:
        ###############################################################################
        # Load Learning to Skim paper's Pickle file
        ###############################################################################
        train_d, dev_d, test_d = helper.get_splited_imdb_data(args.output_base_path+'IMDB/'+'imdb.p', SAG = args.SAG)
        train_corpus.parse(train_d, task, args.max_example)
        dev_corpus.parse(dev_d, task, args.max_example)
        test_corpus.parse(test_d, task, args.max_example)
    else:
        train_corpus.parse(args.data + task + '/train.txt', task, args.max_example)
        if task == 'multinli':
            dev_corpus.parse(args.data + task + '/dev_matched.txt', task, args.tokenize)
            test_corpus.parse(args.data + task + '/test_matched.txt', task, args.tokenize)
        else:

            dev_corpus.parse(args.data + task + '/dev.txt', task, args.tokenize)
            test_corpus.parse(args.data + task + '/test.txt', task, args.tokenize)



if args.debug:
	threshold_examples = 20
	mid_train = int(len(train_corpus.data)/2)
	mid_dev = int(len(dev_corpus.data)/2)
	mid_test = int(len(test_corpus.data)/2)
	train_corpus.data = train_corpus.data[mid_train-threshold_examples:mid_train+threshold_examples]
	dev_corpus.data = dev_corpus.data[mid_dev-threshold_examples:mid_dev+threshold_examples]
	test_corpus.data = test_corpus.data[mid_test-threshold_examples:mid_test+threshold_examples]



print('train set size = ', len(train_corpus.data))
print('development set size = ', len(dev_corpus.data))
print('test set size = ', len(test_corpus.data))


# save the dictionary object to use during testing
if os.path.exists(args.output_base_path + args.task+'/'+'dictionary.p'):
    print('loading dictionary')
    dictionary = helper.load_object(args.output_base_path + args.task+'/'+ 'dictionary.p') 
else:
    dictionary = data.Dictionary()
    dictionary.build_dict(train_corpus.data, args.max_words) # limitation trained L1 models are on all vocab (train , test, dev) as they are all loaded
    helper.save_object(dictionary, args.output_base_path + args.task+'/' + 'dictionary.p')


    
print('vocabulary size = ', len(dictionary))



# ###############################################################################

# train = train.Train(model, optimizer, selector, optimizer_selector, dictionary, args, best_acc)
# train.train_epochs(train_corpus, dev_corpus, test_corpus, args.start_epoch, args.epochs)



numpy.random.shuffle(train_corpus.data)#helper.batchify(train_corpus.data, args.batch_size)

# num_batches=len(train_batches)


i = 0
# for c in  [0.1, 0.5, 1.0, 5.0, 10, 20, 30, 45, 48, 50, 55, 60, 70, 100, 500, 1000, 5000, 10000, 50000]:
for idx in [0]:
    c = args.c
    print('loading model with c: ', c, ' in iter: ', i+1)# save the model to disk
    filename = args.task+'_L1_model.pcl'

    loaded_model = pickle.load(open(filename, 'rb'))



    time_needed = time.time()
    non_zero_indices = []
    for i in range(len(loaded_model.coef_)): 
        non_zero_indices.append(loaded_model.coef_[i].nonzero()[0])
    non_zero_indices = numpy.concatenate(non_zero_indices, axis=0)
    # import pdb
    # pdb.set_trace()
    time_needed = time.time() - time_needed





    # if 'yelp' not in task:
    save_data_routine( train_corpus, dictionary, non_zero_indices=non_zero_indices, file = 'train_c_'+str(c))
    save_data_routine( dev_corpus, dictionary, non_zero_indices=non_zero_indices, file = 'dev_c_'+str(c))
    save_data_routine( test_corpus, dictionary, non_zero_indices=non_zero_indices, file = 'test_c_'+str(c))

    print(" + commn prepos time neede: ", time_needed)











