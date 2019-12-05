"""
This is the main script for training the model. It loads the datasets, creates the unet model and trains it with the
loaded datasets. After training the model and results of training are saved to files.
"""

import csv

import numpy as np
import os
from Full_vasculature.Utils import config
from Full_vasculature.data_processing.prepare_train_val_sets import create_training_datasets_fold, create_training_datasets
from Full_vasculature.Training.train_function import train_and_save

from Unet.utils.helper import read_tuned_params_from_csv
import pickle


def main(model_def, xval=False):
	
	# fix random seed for reproducibility
	np.random.seed(7)
	# set numpy to print only 3 decimal digits for neatness
	np.set_printoptions(precision=9, suppress=True)

	################################################
	# TRAINING PARAMETERS
	################################################
	metrics = config.METRICS
	optimizer = config.OPTIMIZER
	num_patches = config.NUM_PATCHES
	

	################################################
	# MODEL PARAMETERS
	################################################
	fine_tune = False
	num_epochs = config.NUM_EPOCHS  # number of epochs
	num_channels = config.NUM_CHANNELS
	activation = config.ACTIVATION
	final_activation = config.FINAL_ACTIVATION
	patch_size_list = config.PATCH_SIZES[model_def] 
	patch_size_list_z = config.PATCH_SIZES_Z[model_def] 
	threshold = config.THRESHOLD


	################################################
	# GRID PARAMETERS
	################################################
	learning_rate = 1e-4  # list with learning rates of the optimizer Adam
	batch_size = 64  # list with batch sizes
	num_kernels = [32, 64, 128, 256]
	dropout = 0.1  # percentage of weights to be dropped
	################################################
	

	print('metrics', metrics)
	print('patch size', patch_size_list)
	print('number of epochs', num_epochs)
	print('batch size', batch_size)
	print('learning rate', learning_rate)
	print('dropout rate', dropout)
	print('threshold', threshold)
	print('________________________________________________________________________________')

	if xval:
		# TRAINING LOOPS
		for fold in range(config.XVAL_FOLDS):
	
			# -----------------------------------------------------------
			# LOADING MODEL DATA
			# -----------------------------------------------------------
			train_X_0, train_X_1, train_y_0, train_y_1, val_X_0, val_X_1, val_y_0, val_y_1, = create_training_datasets_fold(patch_size_list, fold)

			# -----------------------------------------------------------
			# TRAIN UNET
			# -----------------------------------------------------------

			# Create folders, files to save
		
			model_path = config.get_model_dir(model_def, fold)
			if not os.path.exists(model_path):
				os.makedirs(model_path)

			model_filepath = config.get_model_filepath(model_def, fold)
			train_metadata_filepath = config.get_train_metadata_filepath(model_def, fold)
			csv_logfile = config.get_train_history_filepath(model_def, fold)

			if os.path.exists(model_filepath):
				print("Trained model exists.")
				continue

			# Run training
			model, history, duration_train = train_and_save(train_X_0, train_X_1, train_y_0, train_y_1, val_X_0, val_X_1, val_y_0, val_y_1,
									patch_size_list, num_epochs, batch_size, learning_rate, dropout, num_channels, num_kernels,
									activation, final_activation, model_def, optimizer, metrics, num_patches,
									threshold, csv_logfile, model_filepath, patch_size_z=patch_size_list_z)

			# -----------------------------------------------------------
			# SAVING MODEL
			# -----------------------------------------------------------
			print('Saving params to ', train_metadata_filepath)
			history.params['batchsize'] = batch_size
			history.params['dropout'] = dropout
			history.params['learning_rate'] = learning_rate
			history.params['samples'] = len(train_X_0[0]) + len(train_X_1[0])
			history.params['val_samples'] = len(val_X_0[0]) + len(val_X_1[0])
			history.params['total_time'] = duration_train
			history.params['model'] = model_def
			results = {'params': history.params, 'history': history.history}
			with open(train_metadata_filepath, 'wb') as handle:
				pickle.dump(results, handle)
	else:
		# -----------------------------------------------------------
		# LOADING MODEL DATA
		# -----------------------------------------------------------
		train_X_0, train_X_1, train_y_0, train_y_1, val_X_0, val_X_1, val_y_0, val_y_1, = create_training_datasets(patch_size_list)

		# -----------------------------------------------------------
		# TRAIN UNET
		# -----------------------------------------------------------

		# Create folders, files to save
		
		model_path = config.get_model_dir(model_def)
		if not os.path.exists(model_path):
			os.makedirs(model_path)

		model_filepath = config.get_model_filepath(model_def)
		train_metadata_filepath = config.get_train_metadata_filepath(model_def)
		csv_logfile = config.get_train_history_filepath(model_def)

		if os.path.exists(model_filepath):
			print("Trained model exists.")
			return

		# Run training
		model, history, duration_train = train_and_save(train_X_0, train_X_1, train_y_0, train_y_1, val_X_0, val_X_1, val_y_0, val_y_1,
								patch_size_list, num_epochs, batch_size, learning_rate, dropout, num_channels, num_kernels,
								activation, final_activation, model_def, optimizer, metrics, num_patches,
								threshold, csv_logfile, model_filepath, patch_size_z=patch_size_list_z)

		# -----------------------------------------------------------
		# SAVING MODEL
		# -----------------------------------------------------------
		print('Saving params to ', train_metadata_filepath)
		history.params['batchsize'] = batch_size
		history.params['dropout'] = dropout
		history.params['learning_rate'] = learning_rate
		history.params['num_kernels'] = num_kernels
		history.params['samples'] = len(train_X_0[0]) + len(train_X_1[0])
		history.params['val_samples'] = len(val_X_0[0]) + len(val_X_1[0])
		history.params['total_time'] = duration_train
		history.params['model'] = model_def
		results = {'params': history.params, 'history': history.history}
		with open(train_metadata_filepath, 'wb') as handle:
			pickle.dump(results, handle)


	print('DONE')

	
if __name__ == '__main__':
	main()