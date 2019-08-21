import os, sys

config_path = 'training_config/'
data_path = '../.allennlp/datasets/'

task = "sst"

# dev_file = '../bcn_output/'+task+'/stop_test.txt'
dev_file = '../bcn_output/'+task+'/test.txt'
# dev_file = '../bcn_output/'+task+'_L1_5.7/test-l1.txt'
# dev_file = '../bcn_output/'+task+'_L1_for_WAG/test-l1.txt'
# dev_file = "../bcn_output/sst_L1_save_as_reported_in_paper/test-l1_speedup1.3x_lstm_c_0.2.txt"
# dev_file = "../bcn_output/sst_L1_save_as_reported_in_paper/test-l1.txt"

server = task+'_skim_RNN_'
# server += 't_1_d_10'
# server = task+'_skim_RNN_d_300'
server += 'WAG'
# server += 'L1'




remove = True
remove = False


option = 'train' 
option = 'evaluate' 
config = config_path + 'skim_RNN_config.json'
# output_path = data_path + server + 'new_WAG'
output_path = data_path + server #+'bad_WAG'
output_path = data_path + 'sst_skim_RNN_t_1_d_10'
model_file = output_path + '/'+'model.tar.gz'
#



run_command = 'python -m allennlp.run ' + option + ' ' 
# run_command = 'python -m allennlp.run ' #+ config + ' ' + output_path + ' ' 
if option == 'train': 
	if remove: os.system('rm -r '+ output_path)
	run_command += config + ' -s ' + ' ' + output_path #+ ' --recover '
# if option == 'evaluate': 
else:
	run_command += model_file + ' '
	run_command += ' --evaluation-data-file '
	run_command += dev_file
	




# print(run_command)
# os.system(run_command)

run_command += ' --theta '
theta=0.45
while  theta < 0.6:
	# in [0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.61, 0.62, 0.63, 0.64]
	run_command_temp = run_command + str(theta)
	remove = False
	print(run_command_temp)
	os.system(run_command_temp)
	theta+=0.01


# for c in [10]: #50000, 5000, 500, 50, 60, 70]:
# 	# dev_file = "../bcn_output/sst_L1/test_c_"+str(c)+"-l1.txt"
# 	if option == 'train': 
# 		if remove: os.system('rm -r '+ output_path)
# 		run_command_tmp = run_command + config + ' -s ' + ' ' + output_path #+ ' --recover '
# 	# if option == 'evaluate': 
# 	else:
# 		run_command_tmp = run_command + model_file + ' '
# 		run_command_tmp +=  ' --evaluation-data-file '
# 		run_command_tmp += dev_file

# 	print(run_command_tmp)
# 	os.system(run_command_tmp)










