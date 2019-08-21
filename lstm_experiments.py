import os, sys

config_path = 'training_config/'
data_path = '../.allennlp/datasets/'

task = 'sst'

dev_file = '../bcn_output/'+task+'/test.txt'
# dev_file = '../bcn_output/'+task+'/stop_test.txt'
# dev_file = '../.allennlp/datasets/sst_test'
# dev_file = '../bcn_output/'+task+'/selected_test.txt'
# dev_file = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt'#data_path + 'squad_dev'

# server = 'yelp_stop_lstm_latest'
server = task+'_lstm'
# server = task+'_lastm_wag_train_without_extra_labels_by_l1s_batch_size_32_both_layer_2'
# server = 'nlp_700_sst'
# server = task+'_stop_lstm_latest'
# server = task+'_lstm_basline_skim_RNN'

remove = True
remove = False

# option = 'predict'
option = 'evaluate'
# option = 'train' 



config = config_path + 'lstm.json'
# config = config_path + 'basline_lstm_skim_RNN.json'




# output_path = data_path + 'agnews_skim_RNN_d_300'# server #+ '_new_WAG'
output_path =data_path + 'sst_skim_RNN_d_300'#_WAG'
# output_path = data_path + 'sst_skim_RNN_WAG'#t_1_d_10'#server +'_WAG'
model_file = output_path + '/'+'model.tar.gz'




run_command = 'python -m allennlp.run ' + option + ' ' 
# run_command = 'python -m allennlp.run ' #+ config + ' ' + output_path + ' ' 
for c in [0.1, 0.5, 1.0, 5.0, 10, 20, 30, 45, 48,50, 100, 1000, 10000]:#[0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.7785, 1.5, 2.5]: #[0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0, 1000.0][4:5]:#
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

	# break

