

import train_unet
import predict_full_brain
import evaluate_segmentation
import compute_error_masks
import pickle

from train_unet import Trainer
from predict_full_brain import Predictor
from evaluate_segmentation import Evaluator

from tensorflow.keras.optimizers import Adam
import os

from prepare_train_val_sets import create_training_datasets
from unet import get_unet_3d, get_context_unet_3d, get_ds_unet_3d, get_brainseg_3d, get_brainseg_3d_2
from unet import get_unet_2d, get_context_unet_2d, get_ds_unet_2d, get_brainseg_2d

from metrics import dice_coef, dice_coef_loss

class Vessel_segmentation():

	def __init__(self, model_def):

		self.TOP_LEVEL = r"D:\\"

		self.DATA_PATH = r"C:\Users\Adam Hilbert\Data_M2\FV_data"
		self.MODEL_PATH = r"C:\Users\Adam Hilbert\Data_M2\FV_data\models\\"
		self.MODEL_DATA_PATH = r"C:\Users\Adam Hilbert\Data_M2\FV_data\model_data\Test"
		self.RESULTS_DIR = r"C:\Users\Adam Hilbert\Data_M2\FV_data\results"

		# ----------------
		# PATCH SETTINGS 
		# ----------------
		self.PATCH_SIZES = {
			'unet-3d': [64],
			'context-unet-3d': [64,128],
			'ds-unet-3d': [64],
			'ds-unet-3d-2': [64],
			'brainseg-3d': [64, 128],
			'brainseg-3d-2': [64, 128],
			'unet-2d': [64],
			'context-unet-2d': [64,128],
			'ds-unet-2d': [64],
			'brainseg-2d': [64, 128]
			}
		  # different quadratic patch sizes n x n
		self.PATCH_SIZES_Z = {
			'unet-3d': [8],
			'context-unet-3d': [8,16],
			'ds-unet-3d': [8],
			'ds-unet-3d-2': [8],
			'brainseg-3d': [8, 16],
			'brainseg-3d-2': [8, 16],
			'unet-2d': [8],
			'context-unet-2d': [8,16],
			'ds-unet-2d': [8],
			'brainseg-2d': [8, 16]
			}  # different quadratic patch sizes n x n
		
		self.NUM_PATCHES = 10  # number of patches we want to extract from one stack (one patient)

		# -----------------------------------------------------------
		# GENERAL MODEL PARAMETERS
		# -----------------------------------------------------------
		self.NUM_CHANNELS = 1  # number of channels of the input images
		self.ACTIVATION = 'relu'  # activation_function after every convolution
		self.FINAL_ACTIVATION = 'sigmoid'  # activation_function of the final layer
		self.LOSS_FUNCTION = dice_coef_loss  # dice loss function defined in Unet/utils/metrics file # 'binary_crossentropy'
		self.METRICS = [dice_coef, 'accuracy']  # , avg_class_acc # dice coefficient defined in Unet/utils/metrics file
		self.OPTIMIZER = Adam  # Adam: algorithm for first-order gradient-based optimization of stochastic objective functions
		self.THRESHOLD = 0.5  # threshold for getting classes from probability maps
		self.NUM_EPOCHS = 2
		self.DROPOUT = 0.1
		self.NUM_KERNELS = [32,64,128,256]

		self.ORIGINAL_DATA_DIR = {'all': os.path.join(self.DATA_PATH, 'original_data', 'all'),
							 'train': os.path.join(self.DATA_PATH, 'original_data', 'train'),
		#					 'test': os.path.join(self.DATA_PATH, 'original_data', 'test'),
							 'val': os.path.join(self.DATA_PATH, 'original_data', 'val')}

		# original files with scans
		self.IMG_FILENAME = '001.nii'
		self.LABEL_FILENAME = '001_Vessel-Manual-Gold-int.nii'

		self.EXECUTABLE_PATH = os.path.join(self.TOP_LEVEL, 'EvaluateSegmentation.exe')
		self.MEASURES = "DICE,JACRD,AUC,KAPPA,RNDIND,ADJRIND,ICCORR,VOLSMTY,MUTINF,HDRFDST@0.95@,AVGDIST,MAHLNBS,VARINFO,GCOERR,PROBDST,SNSVTY,SPCFTY,PRCISON,FMEASR,ACURCY,FALLOUT,TP,FP,TN,FN,REFVOL,SEGVOL"

		self.XVAL_FOLDS = 4

		self.model_def = model_def
		self.model_trained = False

		return None

	# Splits of patients stored in dictionary
	def get_xval_fold_splits_filepath():
		return os.path.join(MODEL_DATA_PATH, "xval_folds.npy")
	def get_probs_path(self, dataset):
		return os.path.join(self.RESULTS_DIR, self.model_def, "probs", dataset)
	def get_errormask_path(self, dataset):
		return os.path.join(self.RESULTS_DIR, self.model_def, "error_masks", dataset)
	def get_eval_path(self, dataset):
		return os.path.join(self.RESULTS_DIR, self.model_def, "eval_segment", dataset)
	

	def get_training_tensors(self, patch_size, num_channels, dropout, num_kernels, patch_size_z=None):
		#### 3D MODELS #####
		if self.model_def == 'unet-3d':
			input_dim = [ [patch_size[0], patch_size[0], patch_size_z[0]] ]

			model = get_unet_3d(input_dim[0], num_channels, dropout, num_kernels=num_kernels)
		
			loss = self.LOSS_FUNCTION
			loss_weights = None
		elif self.model_def == 'context-unet-3d':
			input_dim = [[patch_size[0], patch_size[0], patch_size_z[0]], [patch_size[1], patch_size[1], patch_size_z[1]]]

			model = get_context_unet_3d(input_dim, num_channels, dropout, num_kernels=num_kernels)

			loss = self.LOSS_FUNCTION
			loss_weights = None
		elif self.model_def == 'ds-unet-3d':
			input_dim = [ [patch_size[0], patch_size[0], patch_size_z[0]] ]

			model = get_ds_unet_3d(input_dim[0], num_channels, dropout, num_kernels=num_kernels)

			# Multiple loss
			loss = {}
			for i in range(len(num_kernels) - 1):
				loss["output-" + str(i)] = self.LOSS_FUNCTION
			loss_weights = {}
			for i in range(len(num_kernels) - 1):
				loss_weights["output-" + str(i)] = 0.5 if i == len(num_kernels) - 2 else 0.5 / (len(num_kernels) - 2)

		elif self.model_def == 'brainseg-3d':
			input_dim = [[patch_size[0], patch_size[0], patch_size_z[0]], [patch_size[1], patch_size[1], patch_size_z[1]]]

			model = get_brainseg_3d(input_dim, num_channels, dropout, num_kernels=num_kernels)

			# Multiple loss
			loss = {}
			for i in range(len(num_kernels) - 1):
				loss["output-" + str(i)] = self.LOSS_FUNCTION
			loss_weights = {}
			for i in range(len(num_kernels) - 1):
				loss_weights["output-" + str(i)] = 0.5 if i == len(num_kernels) - 2 else 0.5 / (len(num_kernels) - 2)

		elif self.model_def == 'brainseg-3d-2':
			input_dim = [[patch_size[0], patch_size[0], patch_size_z[0]], [patch_size[1], patch_size[1], patch_size_z[1]]]

			model = get_brainseg_3d_2(input_dim, num_channels, dropout, num_kernels=num_kernels)

			loss = self.LOSS_FUNCTION
			loss_weights = None

		#### 2D MODELS #####
		elif self.model_def == 'unet-2d':
			input_dim = [ [patch_size[0], patch_size[0]] ]

			model = get_unet_2d(input_dim[0], num_channels, dropout, num_kernels=num_kernels)
		
			loss = self.LOSS_FUNCTION
			loss_weights = None
		elif self.model_def == 'context-unet-2d':
			input_dim = [[patch_size[0], patch_size[0]], [patch_size[1], patch_size[1]]]

			model = get_context_unet_2d(input_dim, num_channels, dropout, num_kernels=num_kernels)

			loss = self.LOSS_FUNCTION
			loss_weights = None
		elif self.model_def == 'ds-unet-2d':
			input_dim = [ [patch_size[0], patch_size[0]] ]

			model = get_ds_unet_2d(input_dim[0], num_channels, dropout, num_kernels=num_kernels)

			# Multiple loss
			loss = {}
			for i in range(len(num_kernels) - 1):
				loss["output-" + str(i)] = self.LOSS_FUNCTION
			loss_weights = {}
			for i in range(len(num_kernels) - 1):
				loss_weights["output-" + str(i)] = 0.5 if i == len(num_kernels) - 2 else 0.5 / (len(num_kernels) - 2)

		elif self.model_def == 'brainseg-2d':
			input_dim = [[patch_size[0], patch_size[0]], [patch_size[1], patch_size[1]]]

			model = get_brainseg_2d(input_dim, num_channels, dropout, num_kernels=num_kernels)

			# Multiple loss
			loss = {}
			for i in range(len(num_kernels) - 1):
				loss["output-" + str(i)] = self.LOSS_FUNCTION
			loss_weights = {}
			for i in range(len(num_kernels) - 1):
				loss_weights["output-" + str(i)] = 0.5 if i == len(num_kernels) - 2 else 0.5 / (len(num_kernels) - 2)
						
		return loss, loss_weights, model, input_dim

	def save_training_meta(self, history):
		# Add model related parameters to meta
		history['params']['model'] = self.model_def
		history['params']['dropout'] = self.DROPOUT

		# Save meta file
		train_metadata_filepath = self.trainer.get_train_metadata_filepath()
		print('Saving params to ', train_metadata_filepath)
		with open(train_metadata_filepath, 'wb') as handle:
			pickle.dump(history, handle)

		return history


	def train(self):
		# -----------------------------------------------------------
		# TRAINING MODEL
		# -----------------------------------------------------------
		
		# Create model
		loss, loss_weights, model, input_dim = self.get_training_tensors(self.PATCH_SIZES[self.model_def], self.NUM_CHANNELS, self.DROPOUT, self.NUM_KERNELS, self.PATCH_SIZES_Z[self.model_def])

		# Create trainer
		self.trainer = Trainer(
							model,
							model_path=os.path.join(self.MODEL_PATH, self.model_def),
							model_data_path=self.MODEL_DATA_PATH,
							metrics = self.METRICS,
							loss = loss,
							loss_weights = loss_weights
				)

		# Train model
		if not self.trainer.model_trained:
			train_metadata = self.trainer.train_model(self.NUM_EPOCHS)
			# Save training meta data
			train_metadata = self.save_training_meta(train_metadata)
			   
		return

	def predict(self, evaluation_dataset):
		# -----------------------------------------------------------
		# MAKE PREDICTIONS
		# -----------------------------------------------------------

		# Retrieve patients
		patients = os.listdir(self.ORIGINAL_DATA_DIR[evaluation_dataset])

		# Create predictor
		self.predictor = Predictor(
							model=self.trainer.model,
							train_metadata=self.trainer.get_train_metadata(), 
							prob_dir=self.get_probs_path(evaluation_dataset),
							error_dir=self.get_errormask_path(evaluation_dataset),
							patients=patients, 
							patients_dir=self.ORIGINAL_DATA_DIR[evaluation_dataset],
							label_filename=self.LABEL_FILENAME,
							threshold=self.THRESHOLD)

		# Retrieve probability masks
		self.predictor.predict_and_save(self.PATCH_SIZES[self.model_def], self.PATCH_SIZES_Z[self.model_def])

		# Retrieve error masks
		self.predictor.make_and_save_error_masks()

	def evaluate(self, evaluation_dataset):
		# -----------------------------------------------------------
		# EVALUATE PREDICTIONS
		# -----------------------------------------------------------

		patients = os.listdir(self.ORIGINAL_DATA_DIR[evaluation_dataset])
		run_params = run_params = {'num epochs': self.NUM_EPOCHS, 'batch size': self.trainer.batch_size,
							'learning rate': self.trainer.learning_rate, 'dropout': self.DROPOUT, 'num_kernels': self.NUM_KERNELS}
		label_files = [os.path.join(self.ORIGINAL_DATA_DIR[evaluation_dataset], patient, self.LABEL_FILENAME) for patient in patients]
		prob_files = [self.predictor.get_probs_filepath(patient) for patient in patients]
		
		self.evaluator = Evaluator(
						patients=patients, 
						run_params=run_params, 
						exec_path=self.EXECUTABLE_PATH, 
						eval_path=self.get_eval_path(evaluation_dataset), 
						label_files=label_files, 
						prob_files=prob_files)

		self.evaluator.evaluate_segmentations(self.THRESHOLD, self.MEASURES)

	def run_pipeline(self, evaluation_dataset):
		
		# Train
		self.train()

		# Predict
		self.predict(evaluation_dataset)

		# Evaluate
		self.evaluate(evaluation_dataset)
		




def main():

	# --- Run training + full brain prediction + evaluation
	models = [
#		"unet-3d",
#		"context-unet-3d",
#		"ds-unet-3d",
		"brainseg-3d"
	]

	for model_def in models:
		print("Model: " + model_def)
		seg = Vessel_segmentation(model_def)
		seg.run_pipeline('val')

#		train_unet.main(model_def, config.PATCH_SIZES[model_def], config.PATCH_SIZES_Z[model_def])
#		predict_full_brain.main(model_def, config.PATCH_SIZES[model_def], config.PATCH_SIZES_Z[model_def])
#		evaluate_segmentation.main(model_def)
#		compute_error_masks.main(model_def)

if __name__ == '__main__':
	main()