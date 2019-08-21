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
    nexamples = len(corpus.data)
    dev_batches = helper.batchify(corpus.data, args.batch_size)
    print('number of train batches = ', len(dev_batches))
    total_acc = 0.0
    correct = 0.0

    num_batches = len(dev_batches)
    n_correct, n_total = 0, 0
    for batch_no in range(1, num_batches + 1):
        if batch_no%500 == 0 : print(' validation batch: ', batch_no, ' of ', num_batches, ' percentage: ', batch_no/num_batches)
        eval_sentences, eval_labels = helper.batch_to_one_hot_encoded(dev_batches[batch_no-1], dictionary, non_zero_indices = non_zero_indices)
        acc = model.score(eval_sentences, eval_labels)
        correct += acc*len(eval_labels)
        total_acc += acc
        # if batch_no%500 == 0 :print(' for this minibatch score: ', acc, ' correct: ', acc*len(eval_labels), ' of ', len(eval_labels), 'total accc: ', total_acc, ' total correct: ', correct)
    print(' Correct: ', correct, ' acc: ', correct/nexamples, ' sanity check: ', total_acc/num_batches)
    return correct/nexamples

def eval(corpus, dictionary, loaded_model, non_zero_indices):
    start = time.time()
    acc = eval_routine(corpus, dictionary, loaded_model, non_zero_indices=non_zero_indices)
    t = time.time() - start
    print('total test time:  %s ' % helper.convert_to_minutes(t))
    return t, acc

def get_trained_model2(c, corpus, dictionary, non_zero_indices):
    model = LogisticRegression(penalty='l1',  tol=0.0001, C=c, fit_intercept=True, \
        intercept_scaling=1, solver='liblinear', max_iter=args.epochs, multi_class='ovr', verbose=0)
    train_batches = helper.batchify(corpus.data, args.batch_size)
    print('number of train batches = ', len(train_batches))

    num_batches = len(train_batches)
    n_correct, n_total = 0, 0
    for batch_no in range(1, num_batches + 1):
        if batch_no%500 == 0 :print(' training batch: ', batch_no, ' of ', num_batches, ' percentage: ', batch_no/num_batches)
        train_sentences1, train_labels = helper.batch_to_one_hot_encoded(train_batches[batch_no-1], dictionary, non_zero_indices = non_zero_indices)
        model.fit(train_sentences1, train_labels)
    return model

def get_trained_model(c, corpus, dictionary, non_zero_indices):
    model = LogisticRegression(penalty='l1',  tol=0.0001, C=c, fit_intercept=True, \
        intercept_scaling=1, solver='liblinear', max_iter=args.epochs, multi_class='ovr', verbose=1)
    train_sentences1, train_labels = helper.batch_to_one_hot_encoded(corpus.data, dictionary, non_zero_indices = non_zero_indices)
    model.fit(train_sentences1, train_labels)
    return model
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
    dictionary.build_dict(train_corpus.data, args.max_words)
    helper.save_object(dictionary, args.output_base_path + args.task+'/' + 'dictionary.p')


    
print('vocabulary size = ', len(dictionary))



# ###############################################################################

# train = train.Train(model, optimizer, selector, optimizer_selector, dictionary, args, best_acc)
# train.train_epochs(train_corpus, dev_corpus, test_corpus, args.start_epoch, args.epochs)



numpy.random.shuffle(train_corpus.data)#helper.batchify(train_corpus.data, args.batch_size)

# num_batches=len(train_batches)



# save the model to disk
filename = args.task+'_emnlp19_L1_model.pcl'
best_acc = 0

for c in  [0.8]:
    print('training model with c: ', c, ' in iter: ', c/.1)
    model = get_trained_model(c, train_corpus, dictionary, non_zero_indices=None)
    # print("==="*20, "\nC: ", c, "\n", "=="*20)
        
    t, score = eval(dev_corpus, dictionary, model, non_zero_indices=None)
    if score > best_acc:
        best_acc = score
        pickle.dump(model, open(filename, 'wb'))
       
 

 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))


non_zero_indices = []
for i in range(len(loaded_model.coef_)): 
    non_zero_indices.append(loaded_model.coef_[i].nonzero()[0])
non_zero_indices = numpy.array(non_zero_indices).flatten()


print('dense model: with c:  ', loaded_model.C)
# td, accd = eval(test_corpus, dictionary, loaded_model, non_zero_indices=None)



print('sparse model: ')
sparse_model = get_trained_model(loaded_model.C, train_corpus, dictionary, non_zero_indices=set(numpy.concatenate( non_zero_indices, axis=0 )))


ts, accs = eval(test_corpus, dictionary, sparse_model, non_zero_indices=set(numpy.concatenate( non_zero_indices, axis=0 )))

print('speed up: ', td/ts, 'tdense: ', td, ' tsparse: ', ts, ' accdense: ', accd , ' accsparse: ', accs)

















