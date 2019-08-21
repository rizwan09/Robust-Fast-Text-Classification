import os, sys

config_path = 'training_config/'
data_path = '../.allennlp/datasets/'
# dev_file = data_path + 'rt_test.txt'
task = 'yelp'
# dev_file = '../bcn_output/'+task+'_L1_works_very_good/test-l1.txt'
# dev_file = '../bcn_output/'+task+'_best_L1/test-l1.txt'
# dev_file = '../bcn_output/'+task+'_L1/test_c_500.0-l1.txt'
# dev_file = '../bcn_output/'+task+'/test.txt'
# dev_file = '../bcn_output/'+task+'/selected_test.txt'
# dev_file = '../.allennlp/datasets/sst_test'
# dev_file = '../bcn_output/sst_L1_save_as_reported_in_paper/test-l1_speedup1.3x_lstm_c_0.2.txt'
# dev_file = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt'#data_path + 'squad_dev'



server = task+'_lstm_L1_6'
server = task+'_lstm_L1_0.01'
server = task+'_lstm_L1_emnlp2019'
# server = task+'_skim_RNN_'



cs = [0.01, 0.05, 0.1, 0.15, 0.25, 0.7785, 1.5]


remove = True
# remove = False
# server = 'sst_lastm_wag_train_without_extra_labels_by_l1s_batch_size_32_both_layer_2'
# option = 'predict'
option = 'evaluate'
option = 'train' 
config = config_path + 'skim_RNN_config.json'
if option == 'evaluate': remove = False
# print(cs)
for c in [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0, 1000.0]: #[0.1, 0.25, 0.5, 0.75, 1.25, 1.5, 1.75]: #[1.0, 2.0, 3.0, 4.0, 5.0, 5.76599128062778, 10.0, 50.0, 100.0, 500.0, 1000.0]:#[0.01, 0.05, 0.1, 0.15, 0.25, 0.7785, 1.5, 2.5]:

	# server = task+'_lstm_L1_'+ str(c)
	# server = task+'_skim_RNN_L1_'+ str(c)
	# print(c)
	# # continue
	# dev_file = '../bcn_output/'+task+'_L1/test_c_'+str(c)+'-l1.txt'
	# dev_file = '../bcn_output/'+task+'_L1/train_c_5.76599128062778-l1.txt'

	output_path = data_path + server #+ 'WAG'

	model_file = output_path + '/' + 'model.tar.gz'




	run_command = 'python -m allennlp.run ' + option + ' ' 
	# run_command = 'python -m allennlp.run ' #+ config + ' ' + output_path + ' ' 
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
	break



