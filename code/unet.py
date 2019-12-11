"""
This file defines the unet architecture.
"""

from tensorflow import split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Convolution2D, Convolution3D, MaxPooling2D, MaxPooling3D, Input, UpSampling2D, UpSampling3D, AveragePooling2D, AveragePooling3D, concatenate, BatchNormalization, Flatten, Dense, Reshape
import numpy as np
from tensorflow.keras.layers import Lambda

def conv_block(m, num_kernels, kernel_size, strides, padding, activation, dropout, data_format, bn):
	"""
	Bulding block with convolutional layers for one level.

	:param m: model
	:param num_kernels: number of convolution filters on the particular level, positive integer
	:param kernel_size: size of the convolution kernel, tuple of two positive integers
	:param strides: strides values, tuple of two positive integers
	:param padding: used padding by convolution, takes values: 'same' or 'valid'
	:param activation: activation_function after every convolution
	:param dropout: percentage of weights to be dropped, float between 0 and 1
	:param data_format: ordering of the dimensions in the inputs, takes values: 'channel_first' or 'channel_last'
	:param bn: weather to use Batch Normalization layers after each convolution layer, True for use Batch Normalization,
	 False do not use Batch Normalization
	:return: model
	"""
	n = Convolution2D(num_kernels, kernel_size, strides=strides, activation=activation, padding=padding,
					  data_format=data_format)(m)
	n = BatchNormalization()(n) if bn else n
	n = Dropout(dropout)(n)
	n = Convolution2D(num_kernels, kernel_size, strides=strides, activation=activation, padding=padding,
					  data_format=data_format)(n)
	n = BatchNormalization()(n) if bn else n
	return n

def conv_block_3d(m, num_kernels, kernel_size, strides, padding, activation, dropout, data_format, bn):
	"""
	Bulding block with convolutional layers for one level.

	:param m: model
	:param num_kernels: number of convolution filters on the particular level, positive integer
	:param kernel_size: size of the convolution kernel, tuple of two positive integers
	:param strides: strides values, tuple of two positive integers
	:param padding: used padding by convolution, takes values: 'same' or 'valid'
	:param activation: activation_function after every convolution
	:param dropout: percentage of weights to be dropped, float between 0 and 1
	:param data_format: ordering of the dimensions in the inputs, takes values: 'channel_first' or 'channel_last'
	:param bn: weather to use Batch Normalization layers after each convolution layer, True for use Batch Normalization,
	 False do not use Batch Normalization
	:return: model
	"""
	n = Convolution3D(num_kernels, kernel_size, strides=strides, activation=activation, padding=padding,
					  data_format=data_format)(m)
	n = BatchNormalization()(n) if bn else n
	n = Dropout(dropout)(n)
	n = Convolution3D(num_kernels, kernel_size, strides=strides, activation=activation, padding=padding,
					  data_format=data_format)(n)
	n = BatchNormalization()(n) if bn else n
	return n

def dense_block(input, depth, data_format):

	n = Dense(depth, activation='sigmoid')(input)
	n = Dense(depth, activation='sigmoid')(n)

	return n

def up_concat_block(m, concat_channels, pool_size, concat_axis, data_format):
	"""
	:param m: model
	:param concat_channels: channels from left side onf Unet to be concatenated with the right part on one level
	:param pool_size: factors by which to downscale (vertical, horizontal), tuple of two positive integers
	:param concat_axis: concatenation axis, concatenate over channels, positive integer
	:param data_format: ordering of the dimensions in the inputs, takes values: 'channel_first' or 'channel_last'
	:return: model    """
	n = UpSampling2D(size=pool_size, data_format=data_format)(m)
	n = concatenate([n, concat_channels], axis=concat_axis)
	return n

def up_concat_block_3d(m, concat_channels, pool_size, concat_axis, data_format):
	"""
	:param m: model
	:param concat_channels: channels from left side onf Unet to be concatenated with the right part on one level
	:param pool_size: factors by which to downscale (vertical, horizontal), tuple of two positive integers
	:param concat_axis: concatenation axis, concatenate over channels, positive integer
	:param data_format: ordering of the dimensions in the inputs, takes values: 'channel_first' or 'channel_last'
	:return: model    """
	n = UpSampling3D(size=pool_size, data_format=data_format)(m)
	n = concatenate([n, concat_channels], axis=concat_axis)
	return n

def down_scale_path(inputs, num_kernels, kernel_size, strides, pool_size, padding, activation, dropout, data_format, bn):
	# DOWN-SAMPLING PART (left side of the U-net)
	# layers on each level: convolution2d -> dropout -> convolution2d -> max-pooling
	# last level without max-pooling
	residuals = {}
	conv_down = inputs
	for i, k in enumerate(num_kernels):
		# level i
		conv_down = conv_block(conv_down, k, kernel_size, strides, padding, activation, dropout, data_format, bn)
		residuals["conv_" + str(i)] = conv_down
		if i < len(num_kernels)-1:
			conv_down = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_down)
				
	return conv_down, residuals

def down_scale_path_3d(inputs, num_kernels, kernel_size, strides, pool_size, padding, activation, dropout, data_format, bn):
	# DOWN-SAMPLING PART (left side of the U-net)
	# layers on each level: convolution2d -> dropout -> convolution2d -> max-pooling
	# last level without max-pooling

	residuals = {}
	conv_down = inputs
	for i, k in enumerate(num_kernels):
		# level i
		conv_down = conv_block_3d(conv_down, k, kernel_size, strides, padding, activation, dropout, data_format, bn)
		residuals["conv_" + str(i)] = conv_down
		if i < len(num_kernels)-1:
			conv_down = MaxPooling3D(pool_size=pool_size, data_format=data_format)(conv_down)
				
	return conv_down, residuals

def up_scale_path(inputs, residuals, num_kernels, kernel_size, strides, pool_size, concat_axis, padding, activation, final_activation, dropout, data_format, bn):

	# UP-SAMPLING PART (right side of the U-net)
	# layers on each level: upsampling2d -> concatenation with feature maps of corresponding level from down-sampling
	# part -> convolution2d -> dropout -> convolution2d
	# final convolutional layer maps feature maps to desired number of classes

	conv_up = inputs
	output_dim = residuals["conv_0"].shape
	for i in range(len(num_kernels)-1):
		# level i
		conv_up = up_concat_block(conv_up, residuals["conv_"+str(len(num_kernels)-i-2)], pool_size, concat_axis, data_format)
		conv_up = conv_block(conv_up, num_kernels[len(num_kernels)-i-2], kernel_size, strides, padding, activation, dropout, data_format, bn)
		
	final_conv = Convolution2D(1, 1, strides=strides, activation=final_activation, padding=padding,
							   data_format=data_format)(conv_up)	
	return final_conv

def up_scale_path_3d(inputs, residuals, num_kernels, kernel_size, strides, pool_size, concat_axis, padding, activation, final_activation, dropout, data_format, bn):

	# UP-SAMPLING PART (right side of the U-net)
	# layers on each level: upsampling2d -> concatenation with feature maps of corresponding level from down-sampling
	# part -> convolution2d -> dropout -> convolution2d
	# final convolutional layer maps feature maps to desired number of classes

	conv_up = inputs
	output_dim = residuals["conv_0"].shape
	for i in range(len(num_kernels)-1):
		# level i
		conv_up = up_concat_block_3d(conv_up, residuals["conv_"+str(len(num_kernels)-i-2)], pool_size, concat_axis, data_format)
		conv_up = conv_block_3d(conv_up, num_kernels[len(num_kernels)-i-2], kernel_size, strides, padding, activation, dropout, data_format, bn)
		
	final_conv = Convolution3D(1, 1, strides=strides, activation=final_activation, padding=padding,
							   data_format=data_format)(conv_up)	
	return final_conv

def up_scale_path_ds_3d(inputs, residuals, num_kernels, kernel_size, strides, pool_size, concat_axis, padding, activation, final_activation, dropout, data_format, bn):

	# UP-SAMPLING PART (right side of the U-net)
	# layers on each level: upsampling2d -> concatenation with feature maps of corresponding level from down-sampling
	# part -> convolution2d -> dropout -> convolution2d
	# final convolutional layer maps feature maps to desired number of classes

	outputs = []
	conv_up = inputs
	output_dim = residuals["conv_0"].shape
	for i in range(len(num_kernels)-1):
		# level i
		conv_up = up_concat_block_3d(conv_up, residuals["conv_"+str(len(num_kernels)-i-2)], pool_size, concat_axis, data_format)
		conv_up = conv_block_3d(conv_up, num_kernels[len(num_kernels)-i-2], kernel_size, strides, padding, activation, dropout, data_format, bn)
		
		if i < len(num_kernels)-2:
			# get level prediction
			output = UpSampling3D(size=(int(output_dim[1].value/conv_up.shape[1].value), int(output_dim[2].value/conv_up.shape[2].value), int(output_dim[3].value/conv_up.shape[3].value)), data_format=data_format)(conv_up)
			output = Convolution3D(1, 1, strides=strides, activation=final_activation, padding=padding,
							   data_format=data_format)(output)
			outputs.append(output)

	final_conv = Convolution3D(1, 1, strides=strides, activation=final_activation, padding=padding,
							   data_format=data_format)(conv_up)
	outputs.append(final_conv)

	return outputs

def up_scale_path_ds(inputs, residuals, num_kernels, kernel_size, strides, pool_size, concat_axis, padding, activation, final_activation, dropout, data_format, bn):

	# UP-SAMPLING PART (right side of the U-net)
	# layers on each level: upsampling2d -> concatenation with feature maps of corresponding level from down-sampling
	# part -> convolution2d -> dropout -> convolution2d
	# final convolutional layer maps feature maps to desired number of classes
	outputs = []
	conv_up = inputs
	output_dim = residuals["conv_0"].shape
	for i in range(len(num_kernels)-1):
		# level i
		conv_up = up_concat_block(conv_up, residuals["conv_"+str(len(num_kernels)-i-2)], pool_size, concat_axis, data_format)
		conv_up = conv_block(conv_up, num_kernels[len(num_kernels)-i-2], kernel_size, strides, padding, activation, dropout, data_format, bn)
		
		if i < len(num_kernels)-2:
			# get level prediction
			output = UpSampling2D(size=(int(output_dim[1]//conv_up.shape[1]), int(output_dim[2]//conv_up.shape[2])), data_format=data_format)(conv_up)
			output = Convolution2D(1, 1, strides=strides, activation=final_activation, padding=padding,
							   data_format=data_format)(output)
			outputs.append(output)

	final_conv = Convolution2D(1, 1, strides=strides, activation=final_activation, padding=padding,
							   data_format=data_format)(conv_up)
	outputs.append(final_conv)

	return outputs


### UNET-2D ###
def get_unet_2d(input_dim, num_channels, dropout, activation='relu', final_activation='sigmoid', 
			 kernel_size=(3, 3), pool_size=(2, 2), strides=(1, 1), num_kernels=None, concat_axis=-1,
			 data_format='channels_last', padding='same', bn=True):
	
	# build model
	# specify the input shape
	inputs = Input((input_dim[0], input_dim[1], num_channels))
	# BN for inputs
	layer = BatchNormalization()(inputs)

	# --- Down-scale side
	conv_down, residuals = down_scale_path(layer, num_kernels, kernel_size, strides, pool_size, padding, activation, dropout, data_format, bn)
	
	# Fully connected 1
	fl1 = Convolution2D(num_kernels[-1], (1, 1), strides=(1, 1), padding="same", activation="relu",
					  data_format=data_format)(conv_down)
	# Fully connected 2
	fl2 = Convolution2D(num_kernels[-1], (1, 1), strides=(1, 1), padding="same", activation="relu",
					  data_format=data_format)(fl1)
	
	# --- Up-scale side
	outputs = up_scale_path(fl2, residuals, num_kernels, kernel_size, strides, pool_size, concat_axis, padding, activation, final_activation, dropout, data_format, bn)

	model = Model(inputs=inputs, outputs=outputs)

	# print out model summary to console
	model.summary()

	return model

### UNET-3D ###
def get_unet_3d(input_dim, num_channels, dropout, activation='relu', final_activation='sigmoid', 
			 kernel_size=(3, 3, 3), pool_size=(2, 2, 2), strides=(1, 1, 1), num_kernels=None, concat_axis=-1,
			 data_format='channels_last', padding='same', bn=True):
	
	# build model
	# specify the input shape
	inputs = Input((input_dim[0], input_dim[1], input_dim[2], num_channels))
	# BN for inputs
	layer = BatchNormalization()(inputs)

	# --- Down-scale side
	conv_down, residuals = down_scale_path_3d(layer, num_kernels, kernel_size, strides, pool_size, padding, activation, dropout, data_format, bn)
	
	# Fully connected 1
	fl1 = Convolution3D(num_kernels[-1], (1, 1, 1), strides=(1, 1, 1), padding="same", activation="relu",
					  data_format=data_format)(conv_down)
	# Fully connected 2
	fl2 = Convolution3D(num_kernels[-1], (1, 1, 1), strides=(1, 1, 1), padding="same", activation="relu",
					  data_format=data_format)(fl1)
	
	# --- Up-scale side
	outputs = up_scale_path_3d(fl2, residuals, num_kernels, kernel_size, strides, pool_size, concat_axis, padding, activation, final_activation, dropout, data_format, bn)

	model = Model(inputs=inputs, outputs=outputs)

	# print out model summary to console
	model.summary()

	return model

### CONTEXT-UNET-2D ###
def get_context_unet_2d(input_dim, num_channels, dropout, activation='relu', final_activation='sigmoid', 
			 kernel_size=(3, 3), pool_size=(2, 2), strides=(1, 1), num_kernels=None, concat_axis=-1,
			 data_format='channels_last', padding='same', bn=True):

	### DOWNS-SCALE PATHS
	model_inputs = []
	outputs = []
	residual_list = []
	   	
	# --- Get low resolution patch size
	lri = 0 if input_dim[0][0] > input_dim[1][0] else 1
	hri = 1 if input_dim[0][0] > input_dim[1][0] else 0

	### BUILD FEEDFORWARD
	for i, dim in enumerate(input_dim):
		# specify the input shape
		input = Input((dim[0], dim[1], num_channels))
		model_inputs.append(input)

		# BN for inputs
		input = BatchNormalization()(input)

		# scale down context path
		if i == lri:
			input = AveragePooling2D(pool_size=(2, 2))(input)

		# build down-scale paths
		conv, residuals = down_scale_path(input, num_kernels, kernel_size, strides, pool_size, padding, activation, dropout, data_format, bn)

		outputs.append(conv)
		residual_list.append(residuals)

	### BOTTLENECK
	# Concat feature maps
	concat = concatenate(outputs, axis=-1)
	# Fully connected 1
	fl1 = Convolution2D(num_kernels[-1], (1, 1), strides=(1, 1), padding="same", activation="relu",
					  data_format=data_format)(concat)
	# Fully connected 2
	fl2 = Convolution2D(num_kernels[-1], (1, 1), strides=(1, 1), padding="same", activation="relu",
					  data_format=data_format)(fl1)

	### CONCAT/PREPARE RESIDUALS
	merged_residuals = {}
	for key in residual_list[0].keys():
		merged_residuals[key] = concatenate([residual_list[0][key], residual_list[1][key]], axis=-1)

	### UP-SCALE PATH
	outputs = up_scale_path(fl2, merged_residuals, num_kernels, kernel_size, 
							strides, pool_size, concat_axis, padding, activation, final_activation, dropout, data_format, bn)

	# --- Create model
	model = Model(inputs=model_inputs, outputs=outputs)

	# --- Print out model summary to console
	model.summary()

	return model

### CONTEXT-UNET-3D ###
def get_context_unet_3d(input_dim, num_channels, dropout, activation='relu', final_activation='sigmoid', 
			 kernel_size=(3, 3, 3), pool_size=(2, 2, 2), strides=(1, 1, 1), num_kernels=None, concat_axis=-1,
			 data_format='channels_last', padding='same', bn=True):

	### DOWNS-SCALE PATHS
	model_inputs = []
	outputs = []
	residual_list = []
	   	
	# --- Get low resolution patch size
	lri = 0 if input_dim[0][0] > input_dim[1][0] else 1
	hri = 1 if input_dim[0][0] > input_dim[1][0] else 0

	### BUILD FEEDFORWARD
	for i, dim in enumerate(input_dim):
		# specify the input shape
		input = Input((dim[0], dim[1], dim[2], num_channels))
		model_inputs.append(input)

		# BN for inputs
		input = BatchNormalization()(input)

		# scale down context path
		if i == lri:
			input = AveragePooling3D(pool_size=(2, 2, 2))(input)

		# build down-scale paths
		conv, residuals = down_scale_path_3d(input, num_kernels, kernel_size, strides, pool_size, padding, activation, dropout, data_format, bn)

		outputs.append(conv)
		residual_list.append(residuals)

	### BOTTLENECK
	# Concat feature maps
	concat = concatenate(outputs, axis=-1)
	# Fully connected 1
	fl1 = Convolution3D(num_kernels[-1], (1, 1, 1), strides=(1, 1, 1), padding="same", activation="relu",
					  data_format=data_format)(concat)
	# Fully connected 2
	fl2 = Convolution3D(num_kernels[-1], (1, 1, 1), strides=(1, 1, 1), padding="same", activation="relu",
					  data_format=data_format)(fl1)

	### CONCAT/PREPARE RESIDUALS
	merged_residuals = {}
	for key in residual_list[0].keys():
		merged_residuals[key] = concatenate([residual_list[0][key], residual_list[1][key]], axis=-1)

	### UP-SCALE PATH
	outputs = up_scale_path_3d(fl2, merged_residuals, num_kernels, kernel_size, 
							strides, pool_size, concat_axis, padding, activation, final_activation, dropout, data_format, bn)

	# --- Create model
	model = Model(inputs=model_inputs, outputs=outputs)

	# --- Print out model summary to console
	model.summary()

	return model

### DS-UNET-2D ###
def get_ds_unet_2d(input_dim, num_channels, dropout, activation='relu', final_activation='sigmoid', 
			 kernel_size=(3, 3), pool_size=(2, 2), strides=(1, 1), num_kernels=None, concat_axis=-1,
			 data_format='channels_last', padding='same', bn=True):
	"""
	Defines the architecture of the u-net. Reconstruction of the u-net introduced in: https://arxiv.org/abs/1505.04597

	:param patch_size: height of the patches, positive integer
	:param num_channels: number of channels of the input images, positive integer
	:param activation: activation_function after every convolution
	:param final_activation: activation_function of the final layer
	:param optimizer: optimization algorithm for updating the weights and bias values
	:param learning_rate: learning_rate of the optimizer, float
	:param dropout: percentage of weights to be dropped, float between 0 and 1
	:param loss_function: loss function also known as cost function
	:param metrics: metrics for evaluation of the model performance
	:param kernel_size: size of the convolution kernel, tuple of two positive integers
	:param pool_size: factors by which to downscale (vertical, horizontal), tuple of two positive integers
	:param strides: strides values, tuple of two positive integers
	:param num_kernels: array specifying the number of convolution filters in every level, list of positive integers
		containing value for each level of the model
	:param concat_axis: concatenation axis, concatenate over channels, positive integer
	:param data_format: ordering of the dimensions in the inputs, takes values: 'channel_first' or 'channel_last'
	:param padding: used padding by convolution, takes values: 'same' or 'valid'
	:param bn: weather to use Batch Normalization layers after each convolution layer, True for use Batch Normalization,
	 False do not use Batch Normalization
	:return: compiled u-net model
	"""
	
	# build model
	# specify the input shape
	inputs = Input((input_dim[0], input_dim[1], num_channels))
	# BN for inputs
	layer = BatchNormalization()(inputs)

	# --- Down-scale side
	conv_down, residuals = down_scale_path_3d(layer, num_kernels, kernel_size, strides, pool_size, padding, activation, dropout, data_format, bn)
	
	# Fully connected 1
	fl1 = Convolution2D(num_kernels[-1], (1, 1), strides=(1, 1), padding="same", activation="relu",
					  data_format=data_format)(conv_down)
	# Fully connected 2
	fl2 = Convolution2D(num_kernels[-1], (1, 1), strides=(1, 1), padding="same", activation="relu",
					  data_format=data_format)(fl1)
	
	# --- Up-scale side
	outputs = up_scale_path_ds(fl2, residuals, num_kernels, kernel_size, strides, pool_size, concat_axis, padding, activation, final_activation, dropout, data_format, bn)

	# --- Set names to outputs
	for i in range(len(outputs)):
		naming_layer = Lambda(lambda x: x[0], name="output-"+str(i))
		outputs[i] = naming_layer([outputs[i], inputs])

	model = Model(inputs=inputs, outputs=outputs)

	# print out model summary to console
	model.summary()

	return model

### DS-UNET-3D ###
def get_ds_unet_3d(input_dim, num_channels, dropout, activation='relu', final_activation='sigmoid', 
			 kernel_size=(3, 3, 3), pool_size=(2, 2, 2), strides=(1, 1, 1), num_kernels=None, concat_axis=-1,
			 data_format='channels_last', padding='same', bn=True):
	"""
	Defines the architecture of the u-net. Reconstruction of the u-net introduced in: https://arxiv.org/abs/1505.04597

	:param patch_size: height of the patches, positive integer
	:param num_channels: number of channels of the input images, positive integer
	:param activation: activation_function after every convolution
	:param final_activation: activation_function of the final layer
	:param optimizer: optimization algorithm for updating the weights and bias values
	:param learning_rate: learning_rate of the optimizer, float
	:param dropout: percentage of weights to be dropped, float between 0 and 1
	:param loss_function: loss function also known as cost function
	:param metrics: metrics for evaluation of the model performance
	:param kernel_size: size of the convolution kernel, tuple of two positive integers
	:param pool_size: factors by which to downscale (vertical, horizontal), tuple of two positive integers
	:param strides: strides values, tuple of two positive integers
	:param num_kernels: array specifying the number of convolution filters in every level, list of positive integers
		containing value for each level of the model
	:param concat_axis: concatenation axis, concatenate over channels, positive integer
	:param data_format: ordering of the dimensions in the inputs, takes values: 'channel_first' or 'channel_last'
	:param padding: used padding by convolution, takes values: 'same' or 'valid'
	:param bn: weather to use Batch Normalization layers after each convolution layer, True for use Batch Normalization,
	 False do not use Batch Normalization
	:return: compiled u-net model
	"""
	
	# build model
	# specify the input shape
	inputs = Input((input_dim[0], input_dim[1], input_dim[2], num_channels))
	# BN for inputs
	layer = BatchNormalization()(inputs)

	# --- Down-scale side
	conv_down, residuals = down_scale_path_3d(layer, num_kernels, kernel_size, strides, pool_size, padding, activation, dropout, data_format, bn)
	
	# Fully connected 1
	fl1 = Convolution3D(num_kernels[-1], (1, 1, 1), strides=(1, 1, 1), padding="same", activation="relu",
					  data_format=data_format)(conv_down)
	# Fully connected 2
	fl2 = Convolution3D(num_kernels[-1], (1, 1, 1), strides=(1, 1, 1), padding="same", activation="relu",
					  data_format=data_format)(fl1)
	
	# --- Up-scale side
	outputs = up_scale_path_ds_3d(fl2, residuals, num_kernels, kernel_size, strides, pool_size, concat_axis, padding, activation, final_activation, dropout, data_format, bn)

	# --- Set names to outputs
	for i in range(len(outputs)):
		naming_layer = Lambda(lambda x: x[0], name="output-"+str(i))
		outputs[i] = naming_layer([outputs[i], inputs])

	model = Model(inputs=inputs, outputs=outputs)

	# print out model summary to console
	model.summary()

	return model


### BRAINSEG-2D ###
def get_brainseg_2d(input_dim, num_channels, dropout, activation='relu', final_activation='sigmoid',
			 kernel_size=(3, 3), pool_size=(2, 2), strides=(1, 1), num_kernels=None, concat_axis=-1,
			 data_format='channels_last', padding='same', bn=True):

	### DOWNS-SCALE PATHS
	model_inputs = []
	outputs = []
	residual_list = []
	   	
	# --- Get low resolution patch size
	lri = 0 if input_dim[0][0] > input_dim[1][0] else 1
	hri = 1 if input_dim[0][0] > input_dim[1][0] else 0

	### BUILD FEEDFORWARD
	for i, dim in enumerate(input_dim):
		# specify the input shape
		input = Input((dim[0], dim[1], num_channels))
		model_inputs.append(input)
		
		# take middle slice for 2D
#		shape = input.get_shape().as_list()[:3] + [1]
#		input = Lambda(lambda x: x[0][:,:,:,x[1],:], output_shape=shape)([input, int(dim[2]/2)])
#		splits = split(input, dim[2], 3)

		# BN for inputs
		input = BatchNormalization()(input) #[:,:,:,int(dim[2]/2),:])

		# scale down context path
		if i == lri:
			input = AveragePooling2D(pool_size=(2, 2))(input)

		# build down-scale paths
		conv, residuals = down_scale_path(input, num_kernels, kernel_size, strides, pool_size, padding, activation, dropout, data_format, bn)
		
		outputs.append(conv)
		residual_list.append(residuals)

	### BOTTLENECK
	# Concat feature maps
	concat = concatenate(outputs, axis=-1)
	# Fully connected 1
	fl1 = Convolution2D(num_kernels[-1], (1, 1), strides=(1, 1), padding="same", activation="relu",
					  data_format=data_format)(concat)
	# Fully connected 2
	fl2 = Convolution2D(num_kernels[-1], (1, 1), strides=(1, 1), padding="same", activation="relu",
					  data_format=data_format)(fl1)
	### CONCAT/PREPARE RESIDUALS
	merged_residuals = {}
	for key in residual_list[0].keys():
		merged_residuals[key] = concatenate([residual_list[0][key], residual_list[1][key]], axis=-1)

	### UP-SCALE PATH
	outputs = up_scale_path_ds(fl2, merged_residuals, num_kernels, kernel_size, 
							strides, pool_size, concat_axis, padding, activation, final_activation, dropout, data_format, bn)

	# --- Set names to outputs
	for i in range(len(outputs)):
		naming_layer = Lambda(lambda x: x[0], name="output-"+str(i))
		outputs[i] = naming_layer([outputs[i], model_inputs[0]])

	# --- Create model
	model = Model(inputs=model_inputs, outputs=outputs)

	# --- Print out model summary to console
	model.summary()

	return model

### BRAINSEG-3D ###
def get_brainseg_3d(input_dim, num_channels, dropout, activation='relu', final_activation='sigmoid', 
			 kernel_size=(3, 3, 3), pool_size=(2, 2, 2), strides=(1, 1, 1), num_kernels=None, concat_axis=-1,
			 data_format='channels_last', padding='same', bn=True):

	### DOWNS-SCALE PATHS
	model_inputs = []
	outputs = []
	residual_list = []
	   	
	# --- Get low resolution patch size
	lri = 0 if input_dim[0][0] > input_dim[1][0] else 1
	hri = 1 if input_dim[0][0] > input_dim[1][0] else 0

	### BUILD FEEDFORWARD
	for i, dim in enumerate(input_dim):
		# specify the input shape
		input = Input((dim[0], dim[1], dim[2], num_channels))
		model_inputs.append(input)

		# BN for inputs
		input = BatchNormalization()(input)

		# scale down context path
		if i == lri:
			input = AveragePooling3D(pool_size=(2, 2, 2))(input)

		# build down-scale paths
		conv, residuals = down_scale_path_3d(input, num_kernels, kernel_size, strides, pool_size, padding, activation, dropout, data_format, bn)

		outputs.append(conv)
		residual_list.append(residuals)

	### BOTTLENECK
	# Concat feature maps
	concat = concatenate(outputs, axis=-1)
	# Fully connected 1
	fl1 = Convolution3D(num_kernels[-1], (1, 1, 1), strides=(1, 1, 1), padding="same", activation="relu",
					  data_format=data_format)(concat)
	# Fully connected 2
	fl2 = Convolution3D(num_kernels[-1], (1, 1, 1), strides=(1, 1, 1), padding="same", activation="relu",
					  data_format=data_format)(fl1)

	### CONCAT/PREPARE RESIDUALS
	merged_residuals = {}
	for key in residual_list[0].keys():
		merged_residuals[key] = concatenate([residual_list[0][key], residual_list[1][key]], axis=-1)

	### UP-SCALE PATH
	outputs = up_scale_path_ds_3d(fl2, merged_residuals, num_kernels, kernel_size, 
							strides, pool_size, concat_axis, padding, activation, final_activation, dropout, data_format, bn)

	# --- Set names to outputs + pretend to use distance input so that save doesnt skip it
	for i in range(len(outputs)):
		naming_layer = Lambda(lambda x: x[0], name="output-"+str(i))
		outputs[i] = naming_layer([outputs[i], model_inputs[0]])

	# --- Create model
	model = Model(inputs=model_inputs, outputs=outputs)

	# --- Print out model summary to console
	model.summary()

	return model
