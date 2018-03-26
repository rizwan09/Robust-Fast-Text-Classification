import os


gpu = 0
task = 'IMDB'

dropout_list = [0.2]

write_to_file = 0
test= 1
# max_example = 10000



pyfile = 'imdb_main.py'
pyfile = 'imdb_test.py'
for dropout in dropout_list[::-1]:
	model_file_name = 'full_SAG_enc.pth.tar'

	temp=''


	run_command = ' python3 '+pyfile+' --gpu '+str(gpu)+temp
	if test==1: 
		model_file_name = 'epoch_'+str(epoch)+'_'+model_file_name
		run_command += ' --no_train '
	
	run_command+= ' --save_path '+model_file_name
	if resume==1: run_command+=' --resume '+model_file_name
	
	if pyfile != 'imdb_test.py' and write_to_file ==1:run_command += ' >> ../bcn_output/' + task+'/'+model_file_name+'_output.txt'
	print (run_command)
	os.system(run_command)
	exit()

#python3 main.py --gpu 2 --no_train  --task IMDB --batch_size 32 --dropout 0.3 --num_class 2 --lr 0.001 --num_units 5 --max_norm 3.0 --save_model model_task_IMDB_batch_size_32_dropout_0.3_num_class_2_lr_0.001_num_units_5_best.pth.tar