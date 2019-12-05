"""
This script contains a function that trains a model with given parameters and saves it.
"""


import time
from Full_vasculature.Utils import config
from Full_vasculature.Unet.unet import get_unet_3d, get_context_unet_3d, get_ds_unet_3d, get_brainseg_3d, get_brainseg_3d_2
from Full_vasculature.Unet.unet import get_unet_2d, get_context_unet_2d, get_ds_unet_2d, get_brainseg_2d
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from Unet.utils.metrics import avg_class_acc, dice_coef_loss, dice_coef
import os
import pickle
import numpy as np
from matplotlib import pyplot


class BalancedDataGenerator(Sequence):
	'Generates data for Keras'
	def __init__(self, class0_image_files, class1_image_files, class0_annotation_files, class1_annotation_files, output_dims, 
				 batch_size=32, dim=(32,32), n_channels=1, shuffle=True):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.class0_image_list = class0_image_files
		self.class0_annotation_list = class0_annotation_files
		self.class1_image_list = class1_image_files
		self.class1_annotation_list = class1_annotation_files
		self.n_channels = n_channels
		self.shuffle = shuffle
		self.output_dims = output_dims

		self._data_gen = self.__data_generation # if len(dim[0]) < 3 else self.__data_generation

		
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor((len(self.class0_image_list[0])+len(self.class1_image_list[0])) / self.batch_size))

	def __getitem__(self, index):
		# Generate indexes of the batch
		class0_indexes = self.class0_indexes[index * int(self.batch_size / 2):(index + 1) * int(self.batch_size / 2)]
		class1_indexes = self.class1_indexes[index * int(self.batch_size / 2):(index + 1) * int(self.batch_size / 2)]
		
		# Generate data
		X, y = self._data_gen(class0_indexes, class1_indexes)

		return X, y

	def __load_image(self, file, mean=None, std=None):
		# load from npz
		return np.load(file)['arr_0']
	
	def __load_label(self, file):
		# load from npz
		return np.load(file)['arr_0']


	def on_epoch_end(self):
		self.class0_indexes = np.arange(len(self.class0_image_list[0]))
		self.class1_indexes = np.arange(len(self.class1_image_list[0]))
		if self.shuffle == True:
			np.random.shuffle(self.class0_indexes)
			np.random.shuffle(self.class1_indexes)
		return

	def __data_generation(self, class0_indexes, class1_indexes):
		x = []
		y = []
		for i, d in enumerate(self.dim):
			# Initialization
			x.append(np.empty((self.batch_size, d[0], d[1], d[2], self.n_channels)))
		
		for d in range(self.output_dims):
			y.append(np.empty((self.batch_size, self.dim[0][0], self.dim[0][1], self.dim[0][2], 1)))

		# Generate data
		i = 0
		for (ind_0, ind_1) in zip(class0_indexes, class1_indexes):
			for j in range(len(self.dim)):
				# Store sample
				x[j][i,:,:,:,0] = self.__load_image(self.class0_image_list[j][ind_0])
				x[j][i+1,:,:,:,0] = self.__load_image(self.class1_image_list[j][ind_1])
			for j in range(self.output_dims):
				# Store annotation
				y[j][i,:,:,:,0] = self.__load_label(self.class0_annotation_list[ind_0])
				y[j][i+1,:,:,:,0] = self.__load_label(self.class1_annotation_list[ind_1])
								
			i += 2

		return x, y

	def __data_generation_2d(self, class0_indexes, class1_indexes):
		x = []
		y = []
		for i, d in enumerate(self.dim):
			# Initialization
			x.append(np.empty((self.batch_size, d[0], d[1], self.n_channels)))
		
		for d in range(self.output_dims):
			y.append(np.empty((self.batch_size, self.dim[0][0], self.dim[0][1], 1)))
			
		# Generate data
		i = 0
		for (ind_0, ind_1) in zip(class0_indexes, class1_indexes):
			for j in range(len(self.dim)):
				# Store sample
				x[j][i,:,:,0] = self.__load_image(self.class0_image_list[j][ind_0])
				x[j][i+1,:,:,0] = self.__load_image(self.class1_image_list[j][ind_1])
			for j in range(self.output_dims):
				# Store annotation
				y[j][i,:,:,0] = self.__load_label(self.class0_annotation_list[ind_0])
				y[j][i+1,:,:,0] = self.__load_label(self.class1_annotation_list[ind_1])
								
			i += 2

		return x, y

def get_training_tensors(model_def, patch_size, num_channels, dropout, num_kernels, patch_size_z=None):
	#### 3D MODELS #####
	if model_def == 'unet-3d':
		num_of_outputs = 1
		input_dim = [ [patch_size[0], patch_size[0], patch_size_z[0]] ]

		model = get_unet_3d(input_dim[0], num_channels, dropout, num_kernels=num_kernels)
		
		loss = config.LOSS_FUNCTION
		loss_weights = None
	elif model_def == 'context-unet-3d':
		num_of_outputs = 1
		input_dim = [[patch_size[0], patch_size[0], patch_size_z[0]], [patch_size[1], patch_size[1], patch_size_z[1]]]

		model = get_context_unet_3d(input_dim, num_channels, dropout, num_kernels=num_kernels)

		loss = config.LOSS_FUNCTION
		loss_weights = None
	elif model_def == 'ds-unet-3d':
		num_of_outputs = 3
		input_dim = [ [patch_size[0], patch_size[0], patch_size_z[0]] ]

		model = get_ds_unet_3d(input_dim[0], num_channels, dropout, num_kernels=num_kernels)

		# Multiple loss
		loss = {}
		for i in range(len(num_kernels) - 1):
			loss["output-" + str(i)] = config.LOSS_FUNCTION
		loss_weights = {}
		for i in range(len(num_kernels) - 1):
			loss_weights["output-" + str(i)] = 0.5 if i == len(num_kernels) - 2 else 0.5 / (len(num_kernels) - 2)

	elif model_def == 'brainseg-3d':
		num_of_outputs = 3
		input_dim = [[patch_size[0], patch_size[0], patch_size_z[0]], [patch_size[1], patch_size[1], patch_size_z[1]]]

		model = get_brainseg_3d(input_dim, num_channels, dropout, num_kernels=num_kernels)

		# Multiple loss
		loss = {}
		for i in range(len(num_kernels) - 1):
			loss["output-" + str(i)] = config.LOSS_FUNCTION
		loss_weights = {}
		for i in range(len(num_kernels) - 1):
			loss_weights["output-" + str(i)] = 0.5 if i == len(num_kernels) - 2 else 0.5 / (len(num_kernels) - 2)

	elif model_def == 'brainseg-3d-2':
		num_of_outputs = 1
		input_dim = [[patch_size[0], patch_size[0], patch_size_z[0]], [patch_size[1], patch_size[1], patch_size_z[1]]]

		model = get_brainseg_3d_2(input_dim, num_channels, dropout, num_kernels=num_kernels)

		loss = config.LOSS_FUNCTION
		loss_weights = None

	#### 2D MODELS #####
	elif model_def == 'unet-2d':
		num_of_outputs = 1
		input_dim = [ [patch_size[0], patch_size[0]] ]

		model = get_unet_2d(input_dim[0], num_channels, dropout, num_kernels=num_kernels)
		
		loss = config.LOSS_FUNCTION
		loss_weights = None
	elif model_def == 'context-unet-2d':
		num_of_outputs = 1
		input_dim = [[patch_size[0], patch_size[0]], [patch_size[1], patch_size[1]]]

		model = get_context_unet_2d(input_dim, num_channels, dropout, num_kernels=num_kernels)

		loss = config.LOSS_FUNCTION
		loss_weights = None
	elif model_def == 'ds-unet-2d':
		num_of_outputs = 3
		input_dim = [ [patch_size[0], patch_size[0]] ]

		model = get_ds_unet_2d(input_dim[0], num_channels, dropout, num_kernels=num_kernels)

		# Multiple loss
		loss = {}
		for i in range(len(num_kernels) - 1):
			loss["output-" + str(i)] = config.LOSS_FUNCTION
		loss_weights = {}
		for i in range(len(num_kernels) - 1):
			loss_weights["output-" + str(i)] = 0.5 if i == len(num_kernels) - 2 else 0.5 / (len(num_kernels) - 2)

	elif model_def == 'brainseg-2d':
		num_of_outputs = 3
		input_dim = [[patch_size[0], patch_size[0]], [patch_size[1], patch_size[1]]]

		model = get_brainseg_2d(input_dim, num_channels, dropout, num_kernels=num_kernels)

		# Multiple loss
		loss = {}
		for i in range(len(num_kernels) - 1):
			loss["output-" + str(i)] = config.LOSS_FUNCTION
		loss_weights = {}
		for i in range(len(num_kernels) - 1):
			loss_weights["output-" + str(i)] = 0.5 if i == len(num_kernels) - 2 else 0.5 / (len(num_kernels) - 2)
						
	return loss, loss_weights, model, input_dim, num_of_outputs

def train_and_save(train_X_0, train_X_1, train_y_0, train_y_1, val_X_0, val_X_1, val_y_0, val_y_1, patch_size, num_epochs, batch_size, lr, dropout, num_channels, num_kernels,
				   activation, final_activation, model_def, optimizer, metrics, num_patches, threshold, csv_logfile, model_filepath, patch_size_z=None):
	print('patch size ', patch_size)
	print('number of epochs ', num_epochs)
	print('batch size ', batch_size)
	print('learning rate ', lr)
	print('dropout ', dropout)
	print('model ', model_def)
	print('num_kernels ', num_kernels)
		
	# -----------------------------------------------------------
	# CREATING MODEL
	# -----------------------------------------------------------
	loss, loss_weights, model, input_dim, num_of_outputs = get_training_tensors(model_def, patch_size, num_channels, dropout, num_kernels, patch_size_z)	
	
	# --- Compile model
	model.compile(optimizer=optimizer(lr=lr), loss=loss,
				metrics=metrics, loss_weights=loss_weights)

	# -----------------------------------------------------------
	# CREATING DATA GENERATOR
	# -----------------------------------------------------------			
	train_generator = BalancedDataGenerator(train_X_0, train_X_1, train_y_0[0], train_y_1[0], num_of_outputs,  
												batch_size=batch_size, dim=input_dim)
	val_generator = BalancedDataGenerator(val_X_0, val_X_1, val_y_0[0], val_y_1[0], num_of_outputs, 
												batch_size=batch_size, dim=input_dim)
	
	# -----------------------------------------------------------
	# TRAINING MODEL
	# -----------------------------------------------------------
	start_train = time.time()
	# keras callback for saving the training history to csv file
	csv_logger = CSVLogger(csv_logfile)
	early_stopper = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=3)
	model_checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', save_best_only=True,
								   save_weights_only=False, mode='auto')
	# training
	steps = int(np.floor((len(train_X_0[0])+len(train_X_1[0])) / batch_size))
	history = model.fit_generator(train_generator, validation_data=val_generator,
									steps_per_epoch=steps,
									epochs=num_epochs,
									verbose=2, shuffle=True, callbacks=[csv_logger, early_stopper, model_checkpoint])

	duration_train = int(time.time() - start_train)
	print('training took:', (duration_train // 3600) % 60, 'hours', (duration_train // 60) % 60,
			'minutes', duration_train % 60,
			'seconds')

	return model, history, duration_train

