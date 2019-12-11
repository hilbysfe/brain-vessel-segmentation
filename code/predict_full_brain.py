"""
This is the main script for predicting a segmentation of an input MRA image. Segmentations can be predicted for multiple
models eather on rough grid (the parameters are then read out from the Unet/models/tuned_params.cvs file) or on fine
grid.
"""

import os
import config
from helper import read_tuned_params_from_csv
from predict_function import predict_and_save
from metrics import dice_coef_loss, dice_coef, tversky_coef_loss, weighted_dice_loss
from tensorflow.keras.models import load_model
from scipy.ndimage.filters import convolve
import tensorflow as tf
import pickle
import numpy as np
import helper
import time


class Predictor():

	def __init__(self, model, train_metadata, prob_dir, error_dir, patients, patients_dir, label_filename, threshold=0.5):
		self.model = model
		self.train_metadata = train_metadata
		self.PROB_DIR = prob_dir
		self.ERROR_DIR = error_dir
		self.patients = patients
		self.PATIENTS_DIR = patients_dir
		self.threshold = threshold
		self.label_filename = label_filename

		return

	# where to save probability map from validation as nifti
	def get_probs_filepath(self, patient):
		return os.path.join(self.PROB_DIR, 'probs_' + patient + '_.nii')
	# where to save error mask
	def get_errormasks_filepath(self, patient):
		return os.path.join(self.ERROR_DIR, 'error_mask_' + patient + '_.nii')

		

	def predict(self, patch_size, data_dir, patch_size_z=None):
		print('________________________________________________________________________________')
		print('patient dir:', data_dir)

		# -----------------------------------------------------------
		# LOADING MODEL, IMAGE AND MASK
		# -----------------------------------------------------------	
		print('> Loading image...')
		img_mat = helper.load_nifti_mat_from_file(
			os.path.join(data_dir, '001.nii')).astype(np.float32)
		print('> Loading mask...')
		if not os.path.exists(os.path.join(data_dir, 'mask.nii')):
			avg_mat = convolve(img_mat.astype(dtype=float), np.ones((16,16,16), dtype=float)/4096, mode='constant', cval=0)
			mask_mat = np.where(avg_mat > 10.0, 1, 0)
			helper.create_and_save_nifti(mask_mat, os.path.join(data_dir, 'mask.nii'))
		else:
			mask_mat = helper.load_nifti_mat_from_file(
				os.path.join(data_dir, 'mask.nii'))

		# -----------------------------------------------------------
		# PREDICTION
		# -----------------------------------------------------------
		# the segmentation is going to be saved in this probability matrix
		prob_mat = np.zeros(img_mat.shape, dtype=np.float32)
		x_dim, y_dim, z_dim = prob_mat.shape
	
		# get the x, y and z coordinates where there is brain
		x, y, z = np.where(mask_mat > 0)
		print('x shape:', x.shape)
		print('y shape:', y.shape)
		print('z shape:', z.shape)

		# get the z slices with brain
		z_slices = np.unique(z)

		# start cutting out and predicting the patches
		starttime_total = time.time()

		if '3d' in self.train_metadata['params']['model']:
			x_min = 0 # min(x)
			y_min = 0 # min(y)
			z_min = 0 # min(z)
			x_max = img_mat.shape[0] # max(x)
			y_max = img_mat.shape[1] # max(y)
			z_max = img_mat.shape[2] # max(z)

			num_x_patches = np.int(np.ceil((x_max - x_min) / patch_size[0]))
			num_y_patches = np.int(np.ceil((y_max - y_min) / patch_size[0]))	
			num_z_patches = np.int(np.ceil((z_max - z_min) / patch_size_z[0]))
	
			if num_z_patches*patch_size_z[0] + (np.max(patch_size_z)-np.min(patch_size_z))//2 > img_mat.shape[2]:
				new_z = (num_z_patches-1)*patch_size_z[0] + patch_size_z[0]//2 + np.max(patch_size_z)//2 # so that we can feed sufficient patches
				temp = np.zeros((img_mat.shape[0], img_mat.shape[1], new_z))
				temp[:, :, :img_mat.shape[2]] = img_mat
				temp[:, :, img_mat.shape[2]:] = img_mat[:,:,-(new_z - img_mat.shape[2]):]
				img_mat = temp

			for ix in range(num_x_patches):
				for iy in range(num_y_patches):
					for iz in range(num_z_patches):
						# find the starting and ending x and y coordinates of given patch
						patch_start_x = patch_size[0] * ix
						patch_end_x = patch_size[0] * (ix + 1)
						patch_start_y = patch_size[0] * iy
						patch_end_y = patch_size[0] * (iy + 1)
						patch_start_z = patch_size_z[0] * iz
						patch_end_z = patch_size_z[0] * (iz + 1)
						if patch_end_x > x_max:
							patch_end_x = x_max
						if patch_end_y > y_max:
							patch_end_y = y_max
						if patch_end_z > z_max:
							patch_end_z = z_max

						# find center loc with ref. size
						center_x = patch_start_x + int(patch_size[0]/2)
						center_y = patch_start_y + int(patch_size[0]/2)
						center_z = patch_start_z + int(patch_size_z[0]/2)

						img_patches = []
						for h, size in enumerate(patch_size):
							img_patch = np.zeros((size, size, patch_size_z[h], 1))
							offset_x = 0
							offset_y = 0
							offset_z = 0
						
							# find the starting and ending x and y coordinates of given patch
							img_patch_start_x = center_x - int(size/2)
							img_patch_end_x = center_x + int(size/2)
							img_patch_start_y = center_y - int(size/2)
							img_patch_end_y = center_y + int(size/2)
							img_patch_start_z = center_z - int(patch_size_z[h]/2)
							img_patch_end_z = center_z + int(patch_size_z[h]/2)
												
							if img_patch_end_x > x_max:
								img_patch_end_x = x_max
							if img_patch_end_y > y_max:
								img_patch_end_y = y_max
							if img_patch_start_x < x_min:
								offset_x = x_min - img_patch_start_x
								img_patch_start_x = x_min							
							if img_patch_start_y < y_min:
								offset_y = y_min - img_patch_start_y
								img_patch_start_y = y_min
							if img_patch_start_z < z_min:
								offset_z = z_min - img_patch_start_z
								img_patch_start_z = z_min

							# get the patch with the found coordinates from the image matrix
							img_patch[offset_x : offset_x + (img_patch_end_x-img_patch_start_x), 
									  offset_y : offset_y + (img_patch_end_y-img_patch_start_y),
									  offset_z : offset_z + (img_patch_end_z-img_patch_start_z), 0] \
							= img_mat[img_patch_start_x: img_patch_end_x, img_patch_start_y: img_patch_end_y, img_patch_start_z:img_patch_end_z]
	
							img_patches.append(np.expand_dims(img_patch.astype(np.float32),0))
						
						# predict the patch with the model and save to probability matrix
						prob_mat[patch_start_x: patch_end_x, patch_start_y: patch_end_y, patch_start_z:patch_end_z] = \
								(np.reshape(
									np.squeeze(self.model.predict(img_patches)[-1]),
									(patch_size[0], patch_size[0], patch_size_z[0])
								 ) > self.THRESHOLD).astype(np.uint8) \
						[:patch_end_x-patch_start_x, :patch_end_y-patch_start_y, :patch_end_z-patch_start_z]
		else:
			# proceed slice by slice
			for i in z_slices:
				print('Slice:', i)
				starttime_slice = time.time()
				slice_vox_inds = np.where(z == i)
				# find all x and y coordinates with brain in given slice
				x_in_slice = x[slice_vox_inds]
				y_in_slice = y[slice_vox_inds]
				# find min and max x and y coordinates
				slice_x_min = min(x_in_slice)
				slice_x_max = max(x_in_slice)
				slice_y_min = min(y_in_slice)
				slice_y_max = max(y_in_slice)

				# calculate number of predicted patches in x and y direction in given slice
				if isinstance(patch_size, list):
					num_of_x_patches = np.int(np.ceil((slice_x_max - slice_x_min) / patch_size[0]))
					num_of_y_patches = np.int(np.ceil((slice_y_max - slice_y_min) / patch_size[0]))			
				else:
					num_of_x_patches = np.int(np.ceil((slice_x_max - slice_x_min) / patch_size))
					num_of_y_patches = np.int(np.ceil((slice_y_max - slice_y_min) / patch_size))
				print('num x patches', num_of_x_patches)
				print('num y patches', num_of_y_patches)
			   		 
				for j in range(num_of_x_patches):
					for k in range(num_of_y_patches):
						# find the starting and ending x and y coordinates of given patch
						patch_start_x = slice_x_min + patch_size[0] * j
						patch_end_x = slice_x_min + patch_size[0] * (j + 1)
						patch_start_y = slice_y_min + patch_size[0] * k
						patch_end_y = slice_y_min + patch_size[0] * (k + 1)
						# if the dimensions of the probability matrix are exceeded shift back the last patch
						if patch_end_x > slice_x_max:
							patch_end_x = slice_x_max
						if patch_end_y > slice_y_max:
							patch_end_y = slice_y_max

						# find center loc with ref. size
						center_x = patch_start_x + int(patch_size[0]/2)
						center_y = patch_start_y + int(patch_size[0]/2)

						img_patches = []
						for h, size in enumerate(patch_size):
							img_patch = np.zeros((size, size, 1))
							offset_x = 0
							offset_y = 0
						
							# find the starting and ending x and y coordinates of given patch
							img_patch_start_x = center_x - int(size/2)
							img_patch_end_x = center_x + int(size/2)
							img_patch_start_y = center_y - int(size/2)
							img_patch_end_y = center_y + int(size/2)
												
							if img_patch_end_x > slice_x_max:
								img_patch_end_x = slice_x_max
							if img_patch_end_y > slice_y_max:
								img_patch_end_y = slice_y_max

							if img_patch_start_x < slice_x_min:
								offset_x = slice_x_min - img_patch_start_x
								img_patch_start_x = slice_x_min							
							if img_patch_start_y < slice_y_min:
								offset_y = slice_y_min - img_patch_start_y
								img_patch_start_y = slice_y_min

							# get the patch with the found coordinates from the image matrix
							img_patch[offset_x : offset_x + (img_patch_end_x-img_patch_start_x), 
									  offset_y : offset_y + (img_patch_end_y-img_patch_start_y), 0] \
							= img_mat[img_patch_start_x: img_patch_end_x, img_patch_start_y: img_patch_end_y, i]

							img_patches.append(np.expand_dims(img_patch,0))

						# predict the patch with the model and save to probability matrix
						prob_mat[patch_start_x: patch_end_x, patch_start_y: patch_end_y, i] = (np.reshape(
							np.squeeze(self.model.predict(img_patches)[-1]),
							(patch_size[0], patch_size[0])) > self.THRESHOLD).astype(np.uint8)[:patch_end_x-patch_start_x, :patch_end_y-patch_start_y]

		# how long does the prediction take for a patient
		duration_total = time.time() - starttime_total
		print('prediction in total took:', (duration_total // 3600) % 60, 'hours',
			  (duration_total // 60) % 60, 'minutes',
			  duration_total % 60, 'seconds')

		return prob_mat

	def predict_and_save(self, patch_size, patch_size_z):

		# Create results dir
		if not os.path.exists(self.PROB_DIR):
			os.makedirs(self.PROB_DIR)

		for patient in self.patients:
			if not os.path.exists(self.get_probs_filepath(patient)):
				# predict
				data_dir = os.path.join(self.PATIENTS_DIR, patient)
				prob_mat = self.predict(patch_size, data_dir, patch_size_z)
				# save
				helper.create_and_save_nifti(prob_mat, self.get_probs_filepath(patient))

		return

	def make_error_mask(self, prob_path, ground_truth_path):
		seg_data = helper.load_nifti_mat_from_file(prob_path)
		ground_truth_data = helper.load_nifti_mat_from_file(ground_truth_path)

		error_array = np.zeros(ground_truth_data.shape)

		equal_mask = seg_data == ground_truth_data

		TP = equal_mask + ground_truth_data == 2

		FP = (seg_data > ground_truth_data)
		FN = (seg_data < ground_truth_data)

		# TP = 1-red, FP = 2-green, FN = 3-blue
		error_array = error_array + TP + FP*2 + FN*3

		return error_array

	def make_and_save_error_masks(self):
		
		# Create results dir
		if not os.path.exists(self.ERROR_DIR):
			os.makedirs(self.ERROR_DIR)

		for patient in self.patients:
			if not os.path.exists(self.get_errormasks_filepath(patient)):
				prob_filepath = self.get_probs_filepath(patient)
				if not os.path.exists(prob_filepath):
					print("No probability mask found.")
				else:
					label_path = os.path.join(self.PATIENTS_DIR, patient, self.label_filename)
					output_path = self.get_errormasks_filepath(patient)
					# get mask
					error_mask = self.make_error_mask(prob_filepath, label_path)
					# save
					helper.create_and_save_nifti(error_mask, output_path)
		return

	

def main(model_def, patch_size, patch_size_z, dataset='test', num_channels=1, dropout=0.1, num_kernels=[32,64,128,256], xval=False):
	
	if xval:
		for fold in range(config.XVAL_FOLDS):
			
			# Load meta data	
			train_metadata_filepath = config.get_train_metadata_filepath(model_def, fold)
			with open(train_metadata_filepath, 'rb') as pickle_file:
				train_metadata = pickle.load(pickle_file)

			# Load model	
			model_filepath = config.get_model_filepath(model_def, fold)
			model = load_model(model_filepath, compile=False)

			# Retrieve patients
			patients = np.load(config.get_xval_fold_splits_filepath())[fold][dataset]

			# Create predictor
			predictor = Predictor(
							model=model,
							train_meta_data=train_meta_data, 
							prob_dir=os.path.join(config.RESULTS_DIR, str(fold), model_def, "probs", dataset),
							error_dir=os.path.join(config.RESULTS_DIR, str(fold), model_def, "error_masks", dataset),
							patients=patients, 
							patients_dir=config.ORIGINAL_DATA_DIR['all'])
					
			# Retrieve probability masks
			predictor.predict_and_save(patch_size, patch_size_z)

			# Retrieve error masks
			predictor.make_and_save_error_masks()

	else:		
		# Load meta data
		train_metadata_filepath = config.get_train_metadata_filepath(model_def)
		with open(train_metadata_filepath, 'rb') as pickle_file:
			train_metadata = pickle.load(pickle_file)

		# Load model
		model_filepath = config.get_model_filepath(model_def)
		model = load_model(model_filepath, compile=False)

		# Retrieve patients
		patients = os.listdir(config.ORIGINAL_DATA_DIR[dataset])

		# Create predictor
		predictor = Predictor(
						model=model,
						train_meta_data=train_meta_data, 
						prob_dir=os.path.join(config.RESULTS_DIR, model_def, "probs", dataset),
						error_dir=os.path.join(config.RESULTS_DIR, model_def, "error_masks", dataset),
						patients=patients, 
						patients_dir=config.ORIGINAL_DATA_DIR[dataset])

		# Retrieve probability masks
		predictor.predict_and_save(patch_size, patch_size_z)

		# Retrieve error masks
		predictor.make_and_save_error_masks()

		
	print('DONE')


if __name__ == '__main__':
	main()
