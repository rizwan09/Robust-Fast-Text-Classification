import os


gpu = 2
task = 'IMDB'

dropout = [0.2]

write_to_file = 0
resume=0
# max_example = 10000
save_selection = 0


pyfile = 'imdb_main.py'
# pyfile = 'imdb_main_WAG_full_classifier.py'
# pyfile = 'imdb_test.py'

sparsity_list = [0.00025, 0.0001]#[0.00075,  0.0005]
coherent_list = [2.0]#[1.0]#[2.0]
debug=0

SAG = 0
WAG = 0
load_model = 1# -1 for no load,  0 for load selector only, 1 for load classifier only
classifier_file_name = 'full_WAG_classifier_margo'+'.pth.tar'#'full_ori.pth.tar'
	


for sparsity in sparsity_list:
	for coherent in coherent_list:

		
		# model_file_name = 'modle_sparsity_'+str(sparsity)+'_coherent_'+str(coherent)+'.pth.tar'
		model_file_name = 'model_sparsity_'+str(sparsity)+'_coherent_'+str(coherent)+'.pth.tar'
		
		# model_file_name = 'full_WAG_classifier_margo'+'.pth.tar'#+'_lr_'+str(lr)+'.pth.tar'
		if WAG==1: model_file_name = 'WAG_'+model_file_name
		if save_selection==1: 
			model_file_name = 'modle_sparsity_'+str(sparsity)+'_coherent_'+str(coherent)+'.pth.tar'
			classifier_file_name = model_file_name
			selector_file_name =  model_file_name

		options=' --model_file_name '+model_file_name +' --load_model ' + str(load_model)+\
		' --sparsity ' +str(sparsity)+ ' --coherent '+str(coherent) +' --print_every 100 --plot_every 100'+' --classifier_file_name '+classifier_file_name#+ ' --selector_file_name '+selector_file_name#--batch_size 32 --max_norm 4.9' #+' --lr '+str(lr)#+' --selector_file_name '+selector_file_name+' --classifier_file_name '+classifier_file_name+
		if save_selection==1: 
			options+= ' --save_selection '+ str(save_selection) +' --batch_size 32'
		if debug==1:options+= ' --debug '

		run_command = ' python3 '+pyfile+' --gpu '+str(gpu)+options



		if resume==1: run_command+=' --resume '+model_file_name
		

		if 'test' not in pyfile and write_to_file ==1:run_command += ' >> ../bcn_output/' + task+'/'+model_file_name+'_output.txt'
		print (run_command)
		os.system(run_command)
		# exit()

#python3 main.py --gpu 2 --no_train  --task IMDB --batch_size 32 --dropout 0.3 --num_class 2 --lr 0.001 --num_units 5 --max_norm 3.0 --save_model model_task_IMDB_batch_size_32_dropout_0.3_num_class_2_lr_0.001_num_units_5_best.pth.tar