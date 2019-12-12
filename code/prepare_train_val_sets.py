"""
This helper script loads the patches for every patient in the data sets train and val and stacks them to one matrix
so that they can be fed into the model for training.
"""

import numpy as np
import os

def get_feature_label_set(datadir, patch_size):
	"""
	Loads the patches for every patient in the given data set and stacks them to one matrix.

	:param dataset: String, train or val.
	:param patch_size: Number, for what patch size the data sets shall be created.
	:param num_patches: Number, how many patches of one size were extracted from one patient.
	:param patients: Dictionary with patient names, same structure as defined in config.py.
	:return: X - features, y - labels.
	"""

	# prepare empty matrices for features (X) and labels (y) to store the loaded patches for every patient
	# retrieve patch file names
	X_0 = []
	X_1 = []
	y_0 = []
	y_1 = []
	for p, patch in enumerate(patch_size):
		X_0.append([os.path.join(datadir, filename) for filename in os.listdir(datadir) if '_img_'+str(patch)+'_vessel' in filename])
		X_1.append([os.path.join(datadir, filename) for filename in os.listdir(datadir) if '_img_'+str(patch)+'_nonvessel' in filename])
		y_0.append([os.path.join(datadir, filename) for filename in os.listdir(datadir) if '_label_'+str(patch)+'_vessel' in filename])
		y_1.append([os.path.join(datadir, filename) for filename in os.listdir(datadir) if '_label_'+str(patch)+'_nonvessel' in filename])
		
	return X_0, X_1, y_0, y_1

def create_training_datasets(patch_size, datadir):
	"""
	Gets the patches for every patient in the data sets train and val for given patch size as a matrix. Normalize them
	to have zero mean and unit variance. And returns them as well as the respective mean and standard deviation.

	:param patch_size: Number, for what patch size the data sets shall be created.
	:return: Train feature set, Train label set, Validation feature set, Validation label set, Respective mean,
	Respective standard deviation.
	"""
	train_X_0, train_X_1, train_y_0, train_y_1 = get_feature_label_set(datadir['train'], patch_size)
	val_X_0, val_X_1, val_y_0, val_y_1 = get_feature_label_set(datadir['val'], patch_size)
		
	return train_X_0, train_X_1, train_y_0, train_y_1, val_X_0, val_X_1, val_y_0, val_y_1
