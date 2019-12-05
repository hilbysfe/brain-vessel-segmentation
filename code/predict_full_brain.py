"""
This is the main script for predicting a segmentation of an input MRA image. Segmentations can be predicted for multiple
models eather on rough grid (the parameters are then read out from the Unet/models/tuned_params.cvs file) or on fine
grid.
"""

import os
from Full_vasculature.Utils import config
from Unet.utils.helper import read_tuned_params_from_csv
from Full_vasculature.Unet.predict_function import predict_and_save
from Unet.utils.metrics import dice_coef_loss, dice_coef, tversky_coef_loss, weighted_dice_loss
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle
import numpy as np
from Unet.utils import helper

def main(model_def, xval=False):
	################################################
	# MODEL PARAMETERS
	################################################
	patch_size_list = config.PATCH_SIZES[model_def] 
	patch_size_list_z = config.PATCH_SIZES_Z[model_def]

	dataset = 'test'
	if xval:
		# PREDICTION LOOPS
		for fold in range(config.XVAL_FOLDS):
			model_path = config.get_model_dir(model_def, fold)
			model_filepath = config.get_model_filepath(model_def, fold)
			train_metadata_filepath = config.get_train_metadata_filepath(model_def, fold)
			with open(train_metadata_filepath, 'rb') as pickle_file:
				train_metadata = pickle.load(pickle_file)

			if not os.path.exists(config.get_probs_path(model_def, dataset, fold)):
				os.makedirs(config.get_probs_path_fold(model_def, dataset, fold))

			model = load_model(model_filepath, compile=False)

			patients = np.load(config.get_xval_fold_splits_filepath())[fold][dataset]

			for patient in patients:
				if not os.path.exists(config.get_probs_filepath_fold(patient, model_def, dataset, fold)):
					# predict
					data_dir = os.path.join(config.ORIGINAL_DATA_DIR['all'], patient)
					prob_mat = predict_and_save(patch_size_list, data_dir, model, train_metadata, patch_size_list_z)
					# save
					helper.create_and_save_nifti(prob_mat, config.get_probs_filepath_fold(patient, model_def, dataset, fold))
	else:
		model_path = config.get_model_dir(model_def)
		model_filepath = config.get_model_filepath(model_def)
		train_metadata_filepath = config.get_train_metadata_filepath(model_def)
		with open(train_metadata_filepath, 'rb') as pickle_file:
			train_metadata = pickle.load(pickle_file)

		if not os.path.exists(config.get_probs_path(model_def, dataset)):
			os.makedirs(config.get_probs_path(model_def, dataset))

		model = load_model(model_filepath, compile=False)

		patients = os.listdir(config.ORIGINAL_DATA_DIR[dataset])

		for patient in patients:
			if not os.path.exists(config.get_probs_filepath(patient, model_def, dataset)):
				# predict
				data_dir = os.path.join(config.ORIGINAL_DATA_DIR['all'], patient)
				prob_mat = predict_and_save(patch_size_list, data_dir, model, train_metadata, patch_size_list_z)
				# save
				helper.create_and_save_nifti(prob_mat, config.get_probs_filepath(patient, model_def, dataset))
	print('DONE')


if __name__ == '__main__':
	main()
