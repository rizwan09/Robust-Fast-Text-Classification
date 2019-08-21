# import os

# pyfile='main.py'
# gpu = 0
# dp = 0.2


# # so far best bz-64, earlystop=4

# run_command = 'python3 '+pyfile+ '  --task sst --num_class 2 --gpu '+str(gpu) + ' --batch_size 64 --dropout '+str(dp)+ ' --model_file_name full_ori.pth.tar '+' --early_stop 5'+' --num_units 4' 
# print(run_command)
# os.system(run_command)


import os


gpu = 0
task = 'sst'

dropout = [0.2]

write_to_file = 0
resume=0
# max_example = 10000
save_selection = 0
full_classifier = 1

pyfile = 'rt_main.py'
# pyfile = 'bcn_rt_main.py'
server = 'crf'

# pyfile = 'imdb_test.py'

sparsity_list = [0]#[ 0.000075,  0.00005, 0.000025, 0.00001]#[0.1, 0.25, 0.5, 0.75, 0.025, 0.05, 0.075, 0.00025, 0.00075, 0.0005] # [0.5 , 0.05, 0.02, 0.015, 0.01, 0.001, 0.00075,  0.0005, 0.00025, 0.0001]
coherent_list = [0]#[5.0]
debug=0

SAG = 0
WAG = 1
load_model = 1# -1 for no load,  0 for load selector only, 1 for load classifier only
classifier_file_name = 'full_ori.pth.tar'


nhid= 300
ffnn_dim = 300

for dp in dropout:
	for sparsity in sparsity_list:
		for coherent in coherent_list:

		
			# model_file_name = 'modle_sparsity_'+str(sparsity)+'_coherent_'+str(coherent)+'.pth.tar'
			model_file_name = 'model_sparsity_'+str(sparsity)+'_coherent_'+str(coherent)+'.pth.tar'
			options = ''
			if WAG==1:
				if full_classifier ==1: 
					pyfile = 'imdb_main_WAG_full_classifier.py'
					# model_file_name = 'full_WAG_classifier_crf'+'_dp_'+str(dp)+'.pth.tar'
					model_file_name = 'full_WAG_classifier_'+server+'.pth.tar'
				else:
					# classifier_file_name = 'full_WAG_classifier_crf'+'_dp_'+str(dp)+'.pth.tar'
					classifier_file_name = 'full_WAG_classifier_'+server+'.pth.tar'
					options+=' --classifier_file_name '+classifier_file_name
					model_file_name = 'WAG_'+model_file_name
				options+= ' --WAG '

			elif SAG==1: 
				if full_classifier ==1: 
					pyfile = 'imdb_main_SAG_full_classifier.py'
					model_file_name = 'full_SAG_classifier_crf'+'.pth.tar'
				else:
					classifier_file_name = 'full_SAG_classifier_crf'+'.pth.tar'
					options+=' --classifier_file_name '+classifier_file_name
				options+= ' --SAG '
			

			options+=' --model_file_name '+model_file_name +' --load_model ' + str(load_model)+ ' --num_class 2 --num_units 4 '+\
			' --sparsity ' +str(sparsity)+ ' --coherent '+str(coherent) + ' --task '+task# +' --dropout '+str(dp)+' --print_every 500 --plot_every 500'+\
			# ' --nhid '+ str(nhid) + ' --ffnn_dim '+str(ffnn_dim) #+' --classifier_file_name '+classifier_file_name+ ' --selector_file_name '+selector_file_name#--batch_size 32 --max_norm 4.9' #+' --lr '+str(lr)#+' --selector_file_name '+selector_file_name+' --classifier_file_name '+classifier_file_name+
			
			

			if save_selection==1: 
				if task=='IMDB':model_file_name = 'modle_sparsity_'+str(sparsity)+'_coherent_'+str(coherent)+'.pth.tar'
				classifier_file_name = model_file_name
				selector_file_name =  model_file_name
				options+= ' --save_selection '+ str(save_selection) +' --batch_size 64' + ' --selector_file_name '+selector_file_name
			if load_model>0 and pyfile=='imdb_test.py': classifier_file_name  = model_file_name
			if load_model>-1 and pyfile=='imdb_test.py': selector_file_name  = model_file_name
			
			if load_model>0 : options += ' --classifier_file_name '+classifier_file_name 
			if load_model>-1 and pyfile=='imdb_test.py': options += ' --selector_file_name '+selector_file_name 
			
			if debug==1:options+= ' --debug '
			run_command = ' python3 '+pyfile+' --gpu '+str(gpu)+options
			if resume==1: run_command+=' --resume '+model_file_name
			

			if 'test' not in pyfile and write_to_file ==1:run_command += ' >> ../bcn_output/' + task+'/'+model_file_name+'_output.txt'
			print (run_command)
			os.system(run_command)
			# exit()

#python3 main.py --gpu 2 --no_train  --task IMDB --batch_size 32 --dropout 0.3 --num_class 2 --lr 0.001 --num_units 5 --max_norm 3.0 --save_model model_task_IMDB_batch_size_32_dropout_0.3_num_class_2_lr_0.001_num_units_5_best.pth.tar