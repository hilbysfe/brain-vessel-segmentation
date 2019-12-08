"""
This is the main script for training the model. It loads the datasets, creates the unet model and trains it with the
loaded datasets. After training the model and results of training are saved to files.
"""

import csv

import numpy as np
import os
import config
from prepare_train_val_sets import create_training_datasets_fold, create_training_datasets
from unet import get_unet_3d, get_context_unet_3d, get_ds_unet_3d, get_brainseg_3d, get_brainseg_3d_2
from unet import get_unet_2d, get_context_unet_2d, get_ds_unet_2d, get_brainseg_2d
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from helper import read_tuned_params_from_csv
import pickle

def get_training_tensors(model_def, patch_size, num_channels, dropout, num_kernels, patch_size_z=None):
	#### 3D MODELS #####
	if model_def == 'unet-3d':
		input_dim = [ [patch_size[0], patch_size[0], patch_size_z[0]] ]

		model = get_unet_3d(input_dim[0], num_channels, dropout, num_kernels=num_kernels)
		
		loss = config.LOSS_FUNCTION
		loss_weights = None
	elif model_def == 'context-unet-3d':
		input_dim = [[patch_size[0], patch_size[0], patch_size_z[0]], [patch_size[1], patch_size[1], patch_size_z[1]]]

		model = get_context_unet_3d(input_dim, num_channels, dropout, num_kernels=num_kernels)

		loss = config.LOSS_FUNCTION
		loss_weights = None
	elif model_def == 'ds-unet-3d':
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
		input_dim = [[patch_size[0], patch_size[0], patch_size_z[0]], [patch_size[1], patch_size[1], patch_size_z[1]]]

		model = get_brainseg_3d_2(input_dim, num_channels, dropout, num_kernels=num_kernels)

		loss = config.LOSS_FUNCTION
		loss_weights = None

	#### 2D MODELS #####
	elif model_def == 'unet-2d':
		input_dim = [ [patch_size[0], patch_size[0]] ]

		model = get_unet_2d(input_dim[0], num_channels, dropout, num_kernels=num_kernels)
		
		loss = config.LOSS_FUNCTION
		loss_weights = None
	elif model_def == 'context-unet-2d':
		input_dim = [[patch_size[0], patch_size[0]], [patch_size[1], patch_size[1]]]

		model = get_context_unet_2d(input_dim, num_channels, dropout, num_kernels=num_kernels)

		loss = config.LOSS_FUNCTION
		loss_weights = None
	elif model_def == 'ds-unet-2d':
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


class Trainer():

	def __init__(self, model, model_path, model_data_path, metrics, loss, loss_weights, optimizer=Adam, num_patches=2000, batch_size=64, learning_rate=1e-4):

		self.model = model
		self.MODEL_PATH = model_path
		self.MODEL_DATA_PATH = model_data_path 
		################################################
		# TRAINING PARAMETERS
		################################################
		self.metrics = metrics
		self.optimizer = optimizer
		self.num_patches = num_patches
		self.learning_rate = learning_rate  
		self.batch_size = batch_size 
		self.loss = loss
		self.loss_weights = loss_weights
		################################################
	
		print('metrics', self.metrics)
		print('batch size', self.batch_size)
		print('learning rate', self.learning_rate)
		print('________________________________________________________________________________')
		
		return None

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
		
	# model path
	def get_model_dir(self):
		return self.MODEL_PATH
	# where to store the trained model
	def get_model_filepath(self):
		return os.path.join(self.MODEL_PATH, 'model.h5py')
	# where to store the results of training with parameters and training history
	def get_train_metadata_filepath(self):
		return os.path.join(self.MODEL_PATH, 'train_metadata.pkl')
	# where to store csv file with training history
	def get_train_history_filepath(self):
		return os.path.join(self.MODEL_PATH, 'train_history.csv')
	# model data for train and test sets
	def get_model_data_dir(self):
		return {'train': os.path.join(self.MODEL_DATA_PATH, 'train'), 'test': os.path.join(self.MODEL_DATA_PATH, 'test')}
	
	# train model
	def train_model(self, num_epochs, fine_tune=False):
		
		# --- Check if trained model exists in model path
		model_filepath = self.get_model_filepath()
		if os.path.exists(model_filepath) and not fine_tune:
			print("Trained model exists.")
			return

		# --- Create folders, files to save		
		model_path = self.get_model_dir()
		if not os.path.exists(model_path):
			os.makedirs(model_path)
			
		# --- Load model data
		train_X_0, train_X_1, train_y_0, train_y_1, val_X_0, val_X_1, val_y_0, val_y_1, = create_training_datasets_fold(patch_size_list, fold)

		# --- Compile model
		self.model.compile(optimizer=optimizer(lr=self.learning_rate), loss=self.loss,
					metrics=self.metrics, loss_weights=self.loss_weights)

		# --- Creating generators	
		num_of_outputs = len(self.loss.keys()) if isinstance(self.loss, dict) else 1
		train_generator = BalancedDataGenerator(train_X_0, train_X_1, train_y_0[0], train_y_1[0], num_of_outputs,  
													batch_size=self.batch_size, dim=input_dim)
		val_generator = BalancedDataGenerator(val_X_0, val_X_1, val_y_0[0], val_y_1[0], num_of_outputs, 
													batch_size=self.batch_size, dim=input_dim)
	
		# -----------------------------------------------------------
		# TRAINING MODEL
		# -----------------------------------------------------------
		start_train = time.time()

		csv_logfile = self.get_train_history_filepath()
		csv_logger = CSVLogger(csv_logfile)
		early_stopper = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=3)
		model_checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', save_best_only=True,
									   save_weights_only=False, mode='auto')
		# training
		steps = int(np.floor((len(train_X_0[0])+len(train_X_1[0])) / self.batch_size))
		history = model.fit_generator(
							train_generator,
							validation_data=val_generator,
							steps_per_epoch=steps,
							epochs=num_epochs,
							verbose=2, 
							shuffle=True, 
							callbacks=[csv_logger, early_stopper, model_checkpoint]
				)

		duration_train = int(time.time() - start_train)
		print('training took:', (duration_train // 3600) % 60, 'hours', (duration_train // 60) % 60,
				'minutes', duration_train % 60,
				'seconds')

		# Save parameters
		train_metadata_filepath = self.get_train_metadata_filepath()
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

		return history
	



def main(model_def, patch_size, num_channels, dropout, num_kernels, patch_size_z, xval=False):
	
	# fix random seed for reproducibility
	np.random.seed(7)
	# set numpy to print only 3 decimal digits for neatness
	np.set_printoptions(precision=9, suppress=True)
	   
	if xval:
		# TRAINING LOOPS
		for fold in range(config.XVAL_FOLDS):
	
			# create model
			loss, loss_weights, model, input_dim = get_training_tensors(model_def, patch_size, num_channels, dropout, num_kernels, patch_size_z)

			# create trainer
			trainer = Trainer(
							model,
							model_path=os.path.join(config.MODEL_PATH, str(fold), model_def),
							model_data_path=os.path.join(config.MODEL_DATA_PATH, str(fold)),
							metrics = config.METRICS,
							loss = loss,
							loss_weights = loss_weights
					)

			# train model
			history = trainer.train_model(config.NUM_EPOCHS)

	else:
		# create model
		loss, loss_weights, model, input_dim = get_training_tensors(model_def, patch_size, num_channels, dropout, num_kernels, patch_size_z)

		# create trainer
		trainer = Trainer(
						model,
						model_path=os.path.join(config.MODEL_PATH, model_def),
						model_data_path=config.MODEL_DATA_PATH,
						metrics = config.METRICS,
						loss = loss,
						loss_weights = loss_weights
				)

		# train model
		history = trainer.train_model(config.NUM_EPOCHS)
	
	print('DONE')

	
if __name__ == '__main__':
	main()