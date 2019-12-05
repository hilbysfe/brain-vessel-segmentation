"""
This helper script loads the patches for every patient in the data sets train and val and stacks them to one matrix
so that they can be fed into the model for training.
"""

import numpy as np
import os
from Full_vasculature.Utils import config
from Unet.utils import helper

def get_feature_label_set(dataset, patch_size):
	"""
	Loads the patches for every patient in the given data set and stacks them to one matrix.

	:param dataset: String, train or val.
	:param patch_size: Number, for what patch size the data sets shall be created.
	:param num_patches: Number, how many patches of one size were extracted from one patient.
	:param patients: Dictionary with patient names, same structure as defined in config.py.
	:return: X - features, y - labels.
	"""

	# prepare empty matrices for features (X) and labels (y) to store the loaded patches for every patient
	if isinstance(patch_size, list):
		# retrieve patch file names
		patients = np.unique([f.split('_')[0] for f in os.listdir(config.MODEL_DATA_DIRS[dataset])])
		nr_of_patches = len(os.listdir(config.MODEL_DATA_DIRS[dataset])) // (len(patients)*len(patch_size)*2)
		print(str(nr_of_patches) + " patches found per patient (per patch_size).")
		X_0 = []
		X_1 = []
		y_0 = []
		y_1 = []
		for patient in patients:
			for p, patch in enumerate(patch_size):
				X_0.append([])
				X_1.append([])
				y_0.append([])
				y_1.append([])
				for i in range(nr_of_patches//2):
					X_0[p].append(os.path.join(config.MODEL_DATA_DIRS[dataset], patient+'_img_'+str(patch)+'_vessel_'+str(i)+'.npz'))
					y_0[p].append(os.path.join(config.MODEL_DATA_DIRS[dataset], patient+'_label_'+str(patch)+'_vessel_'+str(i)+'.npz'))
				for i in range(nr_of_patches//2, nr_of_patches):
					X_1[p].append(os.path.join(config.MODEL_DATA_DIRS[dataset], patient+'_img_'+str(patch)+'_nonvessel_'+str(i)+'.npz'))
					y_1[p].append(os.path.join(config.MODEL_DATA_DIRS[dataset], patient+'_label_'+str(patch)+'_nonvessel_'+str(i)+'.npz'))

	else:
		X_0 = [os.path.join(config.MODEL_DATA_DIRS[dataset], file) for file in filenames if 'img_' + str(patch_size) + '_vessel' in file]
		X_1 = [os.path.join(config.MODEL_DATA_DIRS[dataset], file) for file in filenames if 'img_' + str(patch_size) + '_nonvessel' in file]
		y_0 = [os.path.join(config.MODEL_DATA_DIRS[dataset], file) for file in filenames if 'label_' + str(patch_size) + '_vessel' in file]
		y_1 = [os.path.join(config.MODEL_DATA_DIRS[dataset], file) for file in filenames if 'label_' + str(patch_size) + '_nonvessel' in file]


	return X_0, X_1, y_0, y_1

def get_feature_label_set_fold(dataset, patch_size, fold):
	"""
	Loads the patches for every patient in the given data set and stacks them to one matrix.

	:param dataset: String, train or val.
	:param patch_size: Number, for what patch size the data sets shall be created.
	:param num_patches: Number, how many patches of one size were extracted from one patient.
	:param patients: Dictionary with patient names, same structure as defined in config.py.
	:return: X - features, y - labels.
	"""

	data_folder = config.get_model_data_dir_fold(fold)[dataset]
	out_data_folder = config.get_model_data_dir_fold(fold)[dataset].replace('Final-Cross-validation', 'Final-Cross-validation-2d')
	print(out_data_folder)
	# retrieve patch file names
	patients = np.unique([f.split('_')[0] for f in os.listdir(data_folder)])
	
	nr_of_patches = len(os.listdir(data_folder)) // (len(patients)*2*2) # 2*2 for img&label and 2 patch size
	print(str(nr_of_patches) + " patches found per patient (per patch_size).")

	X_0 = []
	X_1 = []
	y_0 = []
	y_1 = []

	# prepare empty matrices for features (X) and labels (y) to store the loaded patches for every patient
	for patient in patients:
		for p, patch in enumerate(patch_size):
			X_0.append([])
			X_1.append([])
			y_0.append([])
			y_1.append([])
			for i in range(nr_of_patches//2):
				# image
				filename = os.path.join(data_folder, patient+'_img_'+str(patch)+'_vessel_'+str(i)+'.npz')
				
#				out_filename = os.path.join(out_data_folder, patient+'_img_'+str(patch)+'_vessel_'+str(i)+'.npz')
#				if not os.path.exists(out_filename):
#					img = np.load(filename)['arr_0'][:,:,(p+1)*4]
#					np.savez_compressed(out_filename, img)
				
				X_0[p].append(filename)

				# label
				filename = os.path.join(data_folder, patient+'_label_'+str(patch)+'_vessel_'+str(i)+'.npz')
				
#				out_filename = os.path.join(out_data_folder, patient+'_label_'+str(patch)+'_vessel_'+str(i)+'.npz')
#				if not os.path.exists(out_filename):
#					img = np.load(filename)['arr_0'][:,:,(p+1)*4]
#					np.savez_compressed(out_filename, img)
				
				y_0[p].append(filename)
			for i in range(nr_of_patches//2, nr_of_patches):
				# image
				filename = os.path.join(data_folder, patient+'_img_'+str(patch)+'_nonvessel_'+str(i)+'.npz')
				
#				out_filename = os.path.join(out_data_folder, patient+'_img_'+str(patch)+'_nonvessel_'+str(i)+'.npz')
#				if not os.path.exists(out_filename):
#					img = np.load(filename)['arr_0'][:,:,(p+1)*4]
#					np.savez_compressed(out_filename, img)
				
				X_1[p].append(filename)
				
				# label				
				filename = os.path.join(data_folder, patient+'_label_'+str(patch)+'_nonvessel_'+str(i)+'.npz')
				
#				out_filename = os.path.join(out_data_folder, patient+'_label_'+str(patch)+'_nonvessel_'+str(i)+'.npz')
#				if not os.path.exists(out_filename):
#					img = np.load(filename)['arr_0'][:,:,(p+1)*4]
#					np.savez_compressed(out_filename, img)

				y_1[p].append(filename)

	return X_0, X_1, y_0, y_1


def create_training_datasets(patch_size):
	"""
	Gets the patches for every patient in the data sets train and val for given patch size as a matrix. Normalize them
	to have zero mean and unit variance. And returns them as well as the respective mean and standard deviation.

	:param patch_size: Number, for what patch size the data sets shall be created.
	:return: Train feature set, Train label set, Validation feature set, Validation label set, Respective mean,
	Respective standard deviation.
	"""
	train_X_0, train_X_1, train_y_0, train_y_1 = get_feature_label_set('train', patch_size)
	val_X_0, val_X_1, val_y_0, val_y_1 = get_feature_label_set('val', patch_size)
		
	return train_X_0, train_X_1, train_y_0, train_y_1, val_X_0, val_X_1, val_y_0, val_y_1

def create_training_datasets_fold(patch_size, fold):
	"""
	Gets the patches for every patient in the data sets train and val for given patch size as a matrix. Normalize them
	to have zero mean and unit variance. And returns them as well as the respective mean and standard deviation.

	:param patch_size: Number, for what patch size the data sets shall be created.
	:return: Train feature set, Train label set, Validation feature set, Validation label set, Respective mean,
	Respective standard deviation.
	"""	
	train_X_0, train_X_1, train_y_0, train_y_1 = get_feature_label_set_fold('train', patch_size, fold)
	val_X_0, val_X_1, val_y_0, val_y_1 = get_feature_label_set_fold('test', patch_size, fold)

	
	return train_X_0, train_X_1, train_y_0, train_y_1, val_X_0, val_X_1, val_y_0, val_y_1
