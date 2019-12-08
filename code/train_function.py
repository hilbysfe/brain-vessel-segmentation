"""
This script contains a function that trains a model with given parameters and saves it.
"""


import time
import config
from unet import get_unet_3d, get_context_unet_3d, get_ds_unet_3d, get_brainseg_3d, get_brainseg_3d_2
from unet import get_unet_2d, get_context_unet_2d, get_ds_unet_2d, get_brainseg_2d
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
import os
import pickle
import numpy as np
from matplotlib import pyplot



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

