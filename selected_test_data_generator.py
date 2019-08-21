import os


gpu = 0
task = 'yelp'

dropout = [0.2]

write_to_file = 0
resume=0
# max_example = 10000
save_selection = 1
full_classifier = 0

pyfile = 'imdb_main.py'



sparsity_list = [5e-05]#[0.0002, 0.00025, 0.0003]#, 0.0005] # [0.5 , 0.05, 0.02, 0.015, 0.01, 0.001, 0.00075,  0.0005, 0.00025, 0.0001]
coherent_list = [8.5]#[1.0, 2.0]
debug=0

SAG = 0
WAG = 1
load_model = 0# -1 for no load,  0 for load selector only, 1 for load classifier only
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
					model_file_name = 'full_WAG_classifier_nlp'+'.pth.tar'
				else:
					# classifier_file_name = 'full_WAG_classifier_nlp'+'.pth.tar'#'full_WAG_classifier_nlp'+'.pth.tar'
					# options+=' --classifier_file_name '+classifier_file_name
					model_file_name = 'WAG_'+model_file_name
				options+= ' --WAG '
			elif SAG==1: 
				if full_classifier ==1: 
					pyfile = 'imdb_main_SAG_full_classifier.py'
					model_file_name = 'full_SAG_classifier_crf'+'.pth.tar'
				else:
					classifier_file_name = 'full_SAG_classifier_nlp'+'.pth.tar'
					options+=' --classifier_file_name '+classifier_file_name
					model_file_name += '_SAG'
				options+= ' --SAG '
			

			options+=' --model_file_name '+model_file_name +' --load_model ' + str(load_model)+ \
			' --sparsity ' +str(sparsity)+ ' --coherent '+str(coherent) # +' --dropout '+str(dp)+' --print_every 500 --plot_every 500'+\
			# ' --nhid '+ str(nhid) + ' --ffnn_dim '+str(ffnn_dim) #+' --classifier_file_name '+classifier_file_name+ ' --selector_file_name '+selector_file_name#--batch_size 32 --max_norm 4.9' #+' --lr '+str(lr)#+' --selector_file_name '+selector_file_name+' --classifier_file_name '+classifier_file_name+
			
			

			if save_selection==1 or pyfile == 'imdb_test.py': 
				# model_file_name = 'modle_sparsity_'+str(sparsity)+'_coherent_'+str(coherent)+'.pth.tar'
				classifier_file_name = model_file_name
				selector_file_name =  model_file_name
				options+= ' --save_selection '+ str(save_selection) +' --batch_size 32'
			if debug==1:options+= ' --debug '
			
			options+=' --selector_file_name '+selector_file_name +' --task '+task + ' --data ../bcn_output/'
			
			run_command = ' python3 '+pyfile+' --gpu '+str(gpu)+options

			if resume==1: run_command+=' --resume '+model_file_name
			

			if 'test' not in pyfile and write_to_file ==1:run_command += ' >> ../bcn_output/' + task+'/'+model_file_name+'_output.txt'
			print (run_command)
			os.system('rm -r ../bcn_output/'+task+'/selected')
			os.system('mkdir ../bcn_output/'+task+'/selected')
			os.system(run_command)
			# exit()

#python3 main.py --gpu 2 --no_train  --task IMDB --batch_size 32 --dropout 0.3 --num_class 2 --lr 0.001 --num_units 5 --max_norm 3.0 --save_model model_task_IMDB_batch_size_32_dropout_0.3_num_class_2_lr_0.001_num_units_5_best.pth.tar001_num_units_5_best.pth.tar