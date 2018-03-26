###############################################################################
# Author: Wasi Ahmad
# Project: Biattentive Classification Network for Sentence Classification
# Date Created: 01/06/2018
#
# File Description: This script tests classification accuracy.
###############################################################################

import torch, helper, util, os, numpy, data, time, pickle, json
from torch import optim
from model import BCN
from selector_model import Selector
from sklearn.metrics import f1_score

args = util.get_args()
# Set the random seed manually for reproducibility.
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)


def evaluate(selector, model, batches, dictionary, outfile=None):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    selector.eval()

    n_correct, n_total = 0, 0
    y_preds, y_true, output = [], [], []
    start = time.time()
    num_batches = len(batches)
    num_tokens = selected_tokens = 0


    for batch_no in range(len(batches)):
        test_sentences1, sent_len1, test_sentences2, sent_len2, test_labels = helper.batch_to_tensors(batches[batch_no],
                                                                                                      dictionary, True)
        if args.cuda:
            test_sentences1 = test_sentences1.cuda()
            test_sentences2 = test_sentences2.cuda()
            test_labels = test_labels.cuda()
        assert test_sentences1.size(0) == test_sentences1.size(0)

        selected_x, sentence1_len, selected_y, sentence2_len, logpz, zsum, zdiff = selector(test_sentences1, sent_len1, test_sentences2, sent_len2)
        score = model(selected_x, sentence1_len, selected_y, sentence2_len)

        preds = torch.max(score, 1)[1]
        if outfile:
            predictions = preds.data.cpu().tolist()
            for i in range(len(batches[batch_no])):
                output.append([batches[batch_no][i].id, predictions[i]])
        else:
            y_preds.extend(preds.data.cpu().tolist())
            y_true.extend(test_labels.data.cpu().tolist())
            n_correct += (preds.view(test_labels.size()).data == test_labels.data).sum()
            n_total += len(batches[batch_no])

            num_tokens += sum(sent_len1)+sum(sent_len2)
            selected_tokens+= sum(sentence1_len)+sum(sentence2_len)


        if (batch_no+1) % args.print_every == 0:
            p = 100.0 * selected_tokens/num_tokens
            print_acc_avg = 100. * n_correct / n_total
            print('%s (%d %d%%) (%.2f) %.2f' % (
                helper.show_progress(start, (batch_no+1) / num_batches), (batch_no+1),
                (batch_no+1) / num_batches * 100, p, print_acc_avg))


    now = time.time()
    s = now - start

    if outfile:
        target_names = ['entailment', 'neutral', 'contradiction']
        with open(outfile, 'w') as f:
            f.write('pairID,gold_label' + '\n')
            for item in output:
                f.write(str(item[0]) + ',' + target_names[item[1]] + '\n')
    else:
        return 100. * n_correct / n_total, 100. * f1_score(numpy.asarray(y_true), numpy.asarray(y_preds),
                                                           average='weighted'), s


if __name__ == "__main__":
    dict_path = model_path = args.save_path 
    if args.full_text: 
        dict_path+=args.task+'/'
        model_path += args.task+'/'
    dict_path += 'dictionary.p'
    if model_path.endswith('.pth.tar')==False: model_path += 'model_best.pth.tar'
    
    dictionary = helper.load_object(dict_path)
    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file,
                                                   dictionary.word2idx)
    model = BCN(dictionary, embeddings_index, args)
    selector = Selector(dictionary, embeddings_index, args)

    if args.cuda:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
        selector = selector.cuda()
    helper.load_selector_classifier_states_from_checkpoint(selector, 'selector', model, model_path, 'state_dict', args.cuda)
    print('vocabulary size = ', len(dictionary))

    task_names = ['snli', 'multinli'] if args.task == 'allnli' else [args.task]
    for task in task_names:
        test_corpus = data.Corpus(args.tokenize)
        if 'IMDB' in args.task:
            ###############################################################################
            # Load Learning to Skim paper's Pickle file
            ###############################################################################
            train_d, dev_d, test_d = helper.get_splited_imdb_data(args.save_path+'data/'+'imdb.p')
            test_corpus.parse(test_d, task, args.max_example)

        elif task == 'multinli' and args.test != 'train':
            for partition in ['_matched', '_mismatched']:
                test_corpus.parse(args.data + task + '/' + args.test + partition + '.txt', task, args.max_example)
                print('[' + partition[1:] + '] dataset size = ', len(test_corpus.data))
                test_batches = helper.batchify(test_corpus.data, args.batch_size)
                if args.test == 'test':
                    evaluate(model, test_batches, dictionary, args.save_path + args.task + partition + '.csv')
                else:
                    test_accuracy, test_f1 = evaluate(model, test_batches, dictionary)
                    print('[' + partition[1:] + '] accuracy: %.2f%%' % test_accuracy)
                    print('[' + partition[1:] + '] f1: %.2f%%' % test_f1)
        else:
            test_corpus.parse(args.data + task + '/' + args.test + '.txt', task, args.max_example)
        print('dataset size = ', len(test_corpus.data))
        test_batches = helper.batchify(test_corpus.data, args.batch_size)
        test_accuracy, test_f1, test_time = evaluate(selector, model, test_batches, dictionary)
        print('accuracy: %.2f%%' % test_accuracy)
        print('f1: %.2f%%' % test_f1)
        print ('test time ', helper.convert_to_minutes(test_time))

