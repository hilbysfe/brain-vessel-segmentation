"""
This file contains helper functions for other scripts.
"""
import csv

import numpy as np
import nibabel as nib
import os

def load_nifti_mat_from_file(path_orig):
	"""
	Loads a nifti file and returns the data from the nifti file as numpy array.
	:param path_orig: String, path from where to load the nifti.
	:return: Nifti data as numpy array.
	"""
	nifti_orig = nib.load(path_orig)
	print(' - nifti loaded from:', path_orig)
	print(' - dimensions of the loaded nifti: ', nifti_orig.shape)
	print(' - nifti data type:', nifti_orig.get_data_dtype())
	return nifti_orig.get_data()  # transform the images into np.ndarrays


def create_and_save_nifti(mat, path_target):
	"""
	Creates a nifti image from numpy array and saves it to given path.
	:param mat: Numpy array.
	:param path_target: String, path where to store the created nifti.
	"""
	new_nifti = nib.Nifti1Image(mat, np.eye(4))  # create new nifti from matrix
	nib.save(new_nifti, path_target)  # save nifti to target dir
	print('New nifti saved to:', path_target)


def read_tuned_params_from_csv(tuned_params_file):
	patch_size_list = []
	num_epochs_list = []
	batch_size_list = []
	learning_rate_list = []
	dropout_list = []

	# read params from csv
	with open(tuned_params_file, 'r') as f:
		reader = csv.reader(f)
		i = 0
		ps_idx = None
		ep_idx = None
		bs_idx = None
		lr_idx = None
		do_idx = None

		for row in reader:
			print(i, ': params', row)

			if i == 0:
				ps_idx = row.index('patch size')
				ep_idx = row.index('num epochs')
				bs_idx = row.index('batch size')
				lr_idx = row.index('learning rate')
				do_idx = row.index('dropout')
			else:
				if len(row) > 0:
					patch_size_list.append(int(row[ps_idx]))
					num_epochs_list.append(int(row[ep_idx]))
					batch_size_list.append(int(row[bs_idx]))
					learning_rate_list.append(float(row[lr_idx]))
					dropout_list.append(float(row[do_idx]))
			i += 1

	return patch_size_list, num_epochs_list, batch_size_list, learning_rate_list, dropout_list
