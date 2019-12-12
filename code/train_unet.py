"""
This is the main script for training the model. It loads the datasets, creates the unet model and trains it with the
loaded datasets. After training the model and results of training are saved to files.
"""

import numpy as np
import os
from prepare_train_val_sets import create_training_datasets
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import pickle
import time


class Trainer():

	def __init__(self, model, model_path, model_data_path, metrics, loss, loss_weights, optimizer, num_patches=10, batch_size=16, learning_rate=1e-4, fine_tune=False):

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
		self.fine_tune = fine_tune
		################################################
	
		print('metrics', self.metrics)
		print('batch size', self.batch_size)
		print('learning rate', self.learning_rate)
		print('________________________________________________________________________________')
		
		self.model_trained = False
		self.train_metadata = None


		# --- Check if trained model exists in model path
		model_filepath = self.get_model_filepath()
		if os.path.exists(model_filepath) and not self.fine_tune:
			print("Trained model exists.")
			self.model_trained = True
			self.model = load_model(model_filepath, compile=False)
			
			with open(self.get_train_metadata_filepath(), 'rb') as pickle_file:
				self.train_metadata = pickle.load(pickle_file)
		
		return

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
		return {'train': os.path.join(self.MODEL_DATA_PATH, 'train'), 'val': os.path.join(self.MODEL_DATA_PATH, 'val'), 'test': os.path.join(self.MODEL_DATA_PATH, 'test')}
	
		
	def get_train_metadata(self):
		return self.train_metadata

	# train model
	def train_model(self, num_epochs):
		
		# --- Check if trained model exists in model path
		if self.model_trained and not self.fine_tune:
			print("Trained model exists.")
			return self.train_meta_data

		# --- Create folders, files to save		
		model_path = self.get_model_dir()
		if not os.path.exists(model_path):
			os.makedirs(model_path)
			
		# --- Load model data
		patch_size_list = [self.model.layers[1].input_shape[0][1], self.model.layers[0].input_shape[0][1]]
		train_X_0, train_X_1, train_y_0, train_y_1, val_X_0, val_X_1, val_y_0, val_y_1, = create_training_datasets(patch_size_list, self.get_model_data_dir())

		# --- Compile model
		self.model.compile(optimizer=self.optimizer(lr=self.learning_rate), loss=self.loss,
					metrics=self.metrics, loss_weights=self.loss_weights)

		# --- Creating generators	
		num_of_outputs = len(self.loss.keys()) if isinstance(self.loss, dict) else 1
		input_dim = [self.model.layers[1].input_shape[0][1:-1], self.model.layers[0].input_shape[0][1:-1]]
		train_generator = self.BalancedDataGenerator(train_X_0, train_X_1, train_y_0[0], train_y_1[0], num_of_outputs,  
													batch_size=self.batch_size, dim=input_dim)
		val_generator = self.BalancedDataGenerator(val_X_0, val_X_1, val_y_0[0], val_y_1[0], num_of_outputs, 
													batch_size=self.batch_size, dim=input_dim)
	
		# -----------------------------------------------------------
		# TRAINING MODEL
		# -----------------------------------------------------------
		start_train = time.time()

		csv_logfile = self.get_train_history_filepath()
		csv_logger = CSVLogger(csv_logfile)
		early_stopper = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=3)
		model_filepath = self.get_model_filepath()
		model_checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', save_best_only=True,
									   save_weights_only=False, mode='auto')
		# training
		steps = int(np.floor((len(train_X_0[0])+len(train_X_1[0])) / self.batch_size))
		history = self.model.fit_generator(
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
				'minutes', duration_train % 60, 'seconds')
			

		# Save parameters		
		history.params['batchsize'] = self.batch_size
		history.params['learning_rate'] = self.learning_rate
		history.params['samples'] = len(train_X_0[0]) + len(train_X_1[0])
		history.params['val_samples'] = len(val_X_0[0]) + len(val_X_1[0])
		history.params['total_time'] = duration_train
		
		self.train_metadata = {'params':history.params, 'history':history.history}
		self.model_trained = True

		return self.train_metadata
	
