import os, sys

config_path = 'training_config/'
data_path = '../.allennlp/datasets/'
# dev_file = data_path + 'rt_test.txt'
# task = 'yelp_tolerence_1'
task = 'yelp'
# dev_file = '../bcn_output/agnews_L1_for_BCN/test-l1.txt'
# dev_file = '../bcn_output/'+task+'_L1_5.7/test-l1.txt'
# dev_file = '../bcn_output/'+task+'_best_L1/test-l1.txt'
# dev_file = '../bcn_output/'+task+'_L1/test_c_7.0-l1.txt'
# dev_file = '../bcn_output/'+task+'_L1/test_c_100.0-l1.txt'
# dev_file = '../bcn_output/'+task+'/test.txt'
# dev_file = '../bcn_output/'+task+'/selected_test.txt'
# dev_file = '../bcn_output/'+task+'/selected/train_selected_sparsity_0.00025_coherent_2.0.txt'
# dev_file = '../bcn_output/'+task+'/stop_test.txt'
# dev_file = '../bcn_output/'+task+'/stop-test.txt'
# dev_file = '../bcn_output/sst_L1_save_as_reported_in_paper/test-l1_speedup1.3x_lstm_c_0.2.txt'
# dev_file = '../.allennlp/datasets/sst_test'
# dev_file = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt'#data_path + 'squad_dev'
server = task
# server = 'agnews_WAG'
# server = 'nlp6_agnews'
# server = 'crf_'+task 
# server = task + '_wag_train_without_extra_labels_by_l1s_dp_0.3_0.2'
# server = task+'_new_BCN_SVM'
# server=IMDB_BCN_L1_c_3.1'
# server=task+'_BCN_L1_6'
# server = 'yelp_tolerence_2'
# server=task+'_BCN_L1_2'
# server=task+'_BCN_stop'
# server = 'nlp_1000_sst_WAG'
# server = 'nlp_sst'#task
# server = 'sst_lastm_wag_train_without_extra_labels_by_l1s_batch_size_32'
# server = task + "_WAG"
# server = task+"_BCN_L1_10"
# server = task +'_BCN_emnlp2019_L1_50'

remove = True
# remove = False

# option = 'predict'
option = 'evaluate'
# option = 'train'
config = config_path + 'sst.json'
# output_path = data_path + server #+ '_WAG_new'
output_path = data_path + server #+ 'bad_WAG'

model_file = output_path + '/'+'model.tar.gz'

if option == 'evaluate': remove = False


run_command = 'python -m allennlp.run ' + option + ' ' 
# run_command = 'python -m allennlp.run ' #+ config + ' ' + output_path + ' ' 
for c in [0.1501]:#[0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0, 1000.0][-2:]:
	dev_file = '../bcn_output/'+task+'_L1/test_c_'+str(c)+'-l1.txt'
	if option == 'evaluate': 
		run_command += model_file + ' '
		run_command += ' --evaluation-data-file '
		run_command += dev_file
		remove = False

	else: 
		if remove: os.system('rm -r '+ output_path)
		run_command += config + ' -s ' + ' ' + output_path #+ ' --recover '


	print(run_command)
	os.system(run_command)

