"""
This file contains helper functions for other scripts.
"""
import csv

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
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


def plot_some_images(mat_list, figure_nr, x, y, title, number_of_rotations=0, slice=60):
	"""
	Helper function to plot some slices from given list of images.
	:param mat_list: List with images as numpy arrays to plot.
	:param figure_nr: Number of pyplot figure.
	:param x: Number, x coordinate of the monitor where the figure will be displayed.
	:param y: Number, y coordinate of the monitor where the figure will be displayed.
	:param title: String, plot title.
	:param number_of_rotations: Number, how many times the image should be rotated by 90° for visualization.
	:param slice: Number of the slice to show.
	:return:
	"""
	print('Plotting...')
	fig = plt.figure(figure_nr)
	fig.canvas.manager.window.move(x, y)
	plt.suptitle(title)
	nr_of_images_to_plot = len(mat_list)
	for i in range(nr_of_images_to_plot):
		# rotate the image data so that the back of the head is down
		rotated_matrix = np.rot90(mat_list[i],
								  number_of_rotations)
		plt.subplot(1, nr_of_images_to_plot, i + 1)
		if rotated_matrix.ndim == 3:
			plt.imshow(rotated_matrix[:, :, slice])
		if rotated_matrix.ndim == 2:
			plt.imshow(rotated_matrix)


def aplly_mask(mat, mask_mat):
	"""
	Masks the image with the given mask.
	:param mat: Numpy array, image to be masked.
	:param mask_mat: Numpy array, mask.
	:return: Numpy array, masked image.
	"""
	masked = mat
	masked[np.where(mask_mat == 0)] = 0
	return masked


def flip(mat, axis):
	"""
	Flips the image over the given axis.
	:param mat: Numpy array, image to be flipped.
	:param axis: Number, Axis over which the image shall be flipped.
	:return: Numpy array, flipped image.
	"""
	return np.flip(mat, axis)


def correct_dimensions(mat_list, slice_dimensions):
	"""
	Checks if all of the matrices in the list have correct given dimensions.
	:param mat_list: List with matrices to check.
	:param slice_dimensions: Tuple of slice dimensions.
	:return: True if dimension correct, False otherwise.
	"""
	correct = True
	for i in range(len(mat_list)):
		if mat_list[i].shape[0] != slice_dimensions[0] or mat_list[i].shape[1] != slice_dimensions[1]:
			correct = False
	return correct


def load_patches(data_dir, patient, patch_size):
	"""
	Loads image and label patches for given patient and patch size from given directory.
	:param data_dir: String, directory where the patches are stored.
	:param patient: String, patient name.
	:param patch_size: Number, patch size.
	:return: Numpy arrays, image and label patches for given patient and given patch size.
	"""
	if isinstance(patch_size, list):
		patch_path = os.path.join(data_dir, str(patient))
	else:
		patch_path = os.path.join(data_dir, 'patch' + str(patch_size), str(patient) + '_' + str(patch_size))
	imgs = np.load(patch_path + '_img.npy')
	labels = np.load(patch_path + '_label.npy')
	return imgs, labels


def plot_loss_acc_history(results, suptitle='', save_name='', val=True, save=True):
	"""
	Plots loss and performance meassure curves from training.
	:param results: Lists of history data for each epoch from the model training.
	:param suptitle: String, main title of the plot.
	:param save_name: String, under what name to save the plot.
	:param val: True if the data from validation shall be plotted as well. False otherwise.
	:param save: True of the plot shall be saved as png image.
	:return:
	"""
	print('Plotting loss and accuracy ...')
	plt.figure(figsize=(10, 5))
	plt.suptitle(suptitle)
	# to make the values on x axis start from 1 and not 0
	x_dim = np.arange(results['params']['epochs'], dtype=int) + 1

	# subplot for plotting the loss values during training
	plt.subplot(1, 2, 1)
	plt.title('train loss')
	plt.plot(x_dim, results['history']['loss'], color='blue', label='train loss')
	if val:
		plt.plot(x_dim, results['history']['val_loss'], color='orange', label='validation loss')
		plt.title('train vs validation loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend()
	plt.grid(True)

	# subplot for plotting the performance measure values during the training
	plt.subplot(1, 2, 2)
	plt.axhline(y=0.9, color='red', linestyle='dotted')
	plt.axhline(y=0.95, color='red', linestyle='dashdot')
	plt.plot(x_dim, results['history']['dice_coef'], color='blue', label='train dice')
	plt.title('train dice')
	if val:
		plt.plot(x_dim, results['history']['val_dice_coef'], color='orange', label='validation dice')
		plt.title('train vs validation dice')
	plt.ylabel('dice')
	plt.xlabel('epoch')
	plt.legend()
	plt.grid(True)

	plt.tight_layout()
	plt.subplots_adjust(top=0.75)
	if save:
		plt.savefig(save_name)


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
