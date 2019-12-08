from Unet.utils import helper
import os
import nibabel as nib
import config
import numpy as np
import matplotlib.pyplot as plt


def make_error_mask(outputpath, prob_path, ground_truth_path):
	seg_data = helper.load_nifti_mat_from_file(prob_path)
	ground_truth_data = helper.load_nifti_mat_from_file(ground_truth_path)

	error_array = np.zeros(ground_truth_data.shape)

	equal_mask = seg_data == ground_truth_data

	TP = equal_mask + ground_truth_data == 2

	FP = (seg_data > ground_truth_data)
	FN = (seg_data < ground_truth_data)

	# TP = 1-red, FP = 2-green, FN = 3-blue
	error_array = error_array + TP + FP*2 + FN*3

	helper.create_and_save_nifti(error_array, outputpath)


def main(model_def, xval=False):
	dataset = 'test'  # train / val / set
	data_dir = config.ORIGINAL_DATA_DIR['all']

	if xval:
		for fold in range(config.XVAL_FOLDS):
			# create results folder for error masks
			if not os.path.exists(config.get_errormasks_path(model_def, dataset, fold)):
				os.makedirs(config.get_errormasks_path(model_def, dataset, fold))
			
			patients = np.load(config.get_xval_fold_splits_filepath())[fold][dataset]
			for patient in patients:
				prob_filepath = config.get_probs_filepath(patient, model_def, dataset, fold)
				if not os.path.exists(prob_filepath):
					print("No probability mask found.")
				else:
					label_path = os.path.join(data_dir, patient, config.LABEL_FILENAME)
					output_path = config.get_errormasks_filepath(patient, model_def, dataset, fold)
					if not os.path.exists(output_path):
						make_error_mask(output_path, prob_filepath, label_path)
	else:
		# create results folder for error masks
		if not os.path.exists(config.get_errormasks_path(model_def, dataset)):
			os.makedirs(config.get_errormasks_path(model_def, dataset))
			
		patients = os.listdir(config.ORIGINAL_DATA[dataset])
		for patient in patients:
			prob_filepath = config.get_probs_filepath(patient, model_def, dataset)
			if not os.path.exists(prob_filepath):
				print("No probability mask found.")
			else:
				label_path = os.path.join(data_dir, patient, config.LABEL_FILENAME)
				output_path = config.get_errormasks_filepath(patient, model_def, dataset)
				if not os.path.exists(output_path):
					make_error_mask(output_path, prob_filepath, label_path)


if __name__ == '__main__':
	main()
