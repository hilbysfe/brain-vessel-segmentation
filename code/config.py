"""
This file contains constants like paths, names, sizes and general parameters of the model which are used in other
scripts.
"""

import os
import time
from metrics import dice_coef_loss, dice_coef
from tensorflow.keras.optimizers import Adam

print(os.getcwd())
TOP_LEVEL = r"/home/adam/code/Vessel_segmentation/"

DATA_PATH = r"C:\Users\Adam Hilbert\Data_M2\FV_data"
#DATA_PATH = r"/data-nvme/adam/"
#MODEL_PATH = r"/data-nvme/adam/models"
MODEL_PATH = r"C:\Users\Adam Hilbert\Data_M2\FV_data\models\\"
EXPERIMENT_NAME = 'Test'
MODEL_DATA_PATH = os.path.join(r"/data-nvme/adam/model_data", EXPERIMENT_NAME)
#MODEL_DATA_PATH = os.path.join(r"C:\Users\Adam Hilbert\Data_M2\FV_data\model_data", EXPERIMENT_NAME)
RESULTS_DIR = os.path.join(r"/data-raid5/adam/results", EXPERIMENT_NAME)

# -----------------------------------------------------------
# PATCH SETTINGS FOR EXTRACTION
# -----------------------------------------------------------
PATCH_SIZES = {
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
PATCH_SIZES_Z = {
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
NUM_PATCHES = 10  # number of patches we want to extract from one stack (one patient)

# -----------------------------------------------------------
# AUGMENTATION SETTINGS
# -----------------------------------------------------------
AUGMENTATION_FACTOR = 1
ZOOM_RANGE = [1.5, 3.0]
CONTRAST_RANGE = [0.25, 3.0]
NOISE_VARIANCE = [0.0, 50.0]

# -----------------------------------------------------------
# GENERAL MODEL PARAMETERS
# -----------------------------------------------------------
NUM_CHANNELS = 1  # number of channels of the input images
ACTIVATION = 'relu'  # activation_function after every convolution
FINAL_ACTIVATION = 'sigmoid'  # activation_function of the final layer
LOSS_FUNCTION = dice_coef_loss  # dice loss function defined in Unet/utils/metrics file # 'binary_crossentropy'
METRICS = [dice_coef, 'accuracy']  # , avg_class_acc # dice coefficient defined in Unet/utils/metrics file
OPTIMIZER = Adam  # Adam: algorithm for first-order gradient-based optimization of stochastic objective functions
THRESHOLD = 0.5  # threshold for getting classes from probability maps
NUM_EPOCHS = 10

# -----------------------------------------------------------
# DIRECTORIES, PATHS AND FILE NAMES
# -----------------------------------------------------------
# directory where the original scans are stored
ORIGINAL_DATA_DIR = {'all': os.path.join(DATA_PATH, 'original_data', 'all')}
#					 'train': os.path.join(DATA_PATH, 'original_data', 'train'),
#					 'test': os.path.join(DATA_PATH, 'original_data', 'test'),
#					 'val': os.path.join(DATA_PATH, 'original_data', 'val')}

# original files with scans
IMG_FILENAME = '001.nii'
LABEL_FILENAME = '001_Vessel-Manual-Gold-int.nii'


# -----------------------------------------------------------
# XVAL RELATED
# -----------------------------------------------------------
XVAL_FOLDS = 4
# Splits of patients stored in dictionary
def get_xval_fold_splits_filepath():
	return os.path.join(MODEL_DATA_PATH, "xval_folds.npy")

# -----------------------------------------------------------
# TRAINING / MODEL FILES
# -----------------------------------------------------------
# model path
def get_model_dir(model_def, fold=None):
	if fold:
		return os.path.join(MODEL_PATH, model_def, str(fold))
	else:
		return os.path.join(MODEL_PATH, model_def)
# model data for train and test sets
def get_model_data_dir(fold=None):
	if fold:
		return {'train': os.path.join(MODEL_DATA_PATH, str(fold), 'train'), 'test': os.path.join(MODEL_DATA_PATH, str(fold), 'test')}
	else:
		return {'train': os.path.join(MODEL_DATA_PATH, 'train'), 'test': os.path.join(MODEL_DATA_PATH, 'test')}
# where to store the trained model
def get_model_filepath(model_def, fold=None):
	if fold:
		return os.path.join(MODEL_PATH, model_def, str(fold), 'model.h5py')
	else:
		return os.path.join(MODEL_PATH, model_def, 'model.h5py')
# where to store the results of training with parameters and training history
def get_train_metadata_filepath(model_def, fold=None):
	if fold:
		return os.path.join(MODEL_PATH, model_def, str(fold), 'train_metadata.pkl')
	else:
		return os.path.join(MODEL_PATH, model_def, 'train_metadata.pkl')
# where to store csv file with training history
def get_train_history_filepath(model_def, fold=None):
	if fold:
		return os.path.join(MODEL_PATH, model_def, str(fold), 'train_history.csv')
	else:
		return os.path.join(MODEL_PATH, model_def, 'train_history.csv')

# -----------------------------------------------------------
# PROBABILITY MASKS
# -----------------------------------------------------------
# where to save probability map from validation as nifti
def get_probs_filepath(patient, model_def, dataset, fold=None):
	if fold:
		return os.path.join(RESULTS_DIR, model_def, str(fold), dataset, 'probs', 'probs_' + patient + '_.nii')
	else:
		return os.path.join(RESULTS_DIR, model_def, dataset, 'probs', 'probs_' + patient + '_.nii')
def get_probs_path(model_def, dataset, fold=None):
	if fold:
		return os.path.join(RESULTS_DIR, model_def, str(fold), dataset, 'probs')
	else:
		return os.path.join(RESULTS_DIR, model_def, dataset, 'probs')

# -----------------------------------------------------------
# ERROR MASKS
# -----------------------------------------------------------

# where to save probability map from validation as nifti
def get_errormasks_filepath(patient, model_def, dataset, fold=None):
	if fold:
		return os.path.join(RESULTS_DIR, model_def, str(fold), dataset, 'error_masks', 'error_mask_' + patient + '_.nii')
	else:
		return os.path.join(RESULTS_DIR, model_def, dataset, 'error_masks', 'error_mask_' + patient + '_.nii')
def get_errormasks_path(model_def, dataset, fold=None):
	if fold:
		return os.path.join(RESULTS_DIR, model_def, str(fold), dataset, 'error_masks')
	else:
		return os.path.join(RESULTS_DIR, model_def, dataset, 'error_masks')


# -----------------------------------------------------------
# EVALUATE SEGMENTATION TOOL
# -----------------------------------------------------------
EXECUTABLE_PATH = TOP_LEVEL + 'EvaluateSegmentation'

def get_eval_segment_dataset_xmlpath(patient, model_def, dataset, fold=None):
	if fold:
		return os.path.join(RESULTS_DIR, model_def, str(fold), dataset, 'eval_segment', 'eval_segment_' + patient + '.xml')
	else:
		return os.path.join(RESULTS_DIR, model_def, dataset, 'eval_segment', 'eval_segment_' + patient + '.xml')
def get_eval_segment_dataset_path(model_def, dataset, fold=None):
	if fold:
		return os.path.join(RESULTS_DIR, model_def, str(fold), dataset, 'eval_segment')
	else:
		return os.path.join(RESULTS_DIR, model_def, dataset, 'eval_segment')

def get_eval_segment_dataset_csvpath_per_patient(model_def, dataset, fold=None):
	if fold:
		return os.path.join(RESULTS_DIR, model_def, str(fold), dataset, 'eval_segment', 'eval_segment_per_patient_' + time.strftime("%Y%m%d-%H%M%S") + '.csv')
	else:
		return os.path.join(RESULTS_DIR, model_def, dataset, 'eval_segment', 'eval_segment_per_patient_' + time.strftime("%Y%m%d-%H%M%S") + '.csv')
def get_eval_segment_dataset_csvpath(model_def, dataset, fold=None):
	if fold:
		return os.path.join(RESULTS_DIR, model_def, str(fold), dataset, 'eval_segment', 'eval_segment_' + time.strftime("%Y%m%d-%H%M%S") + '.csv')
	else:
		return os.path.join(RESULTS_DIR, model_def, dataset, 'eval_segment', 'eval_segment_' + time.strftime("%Y%m%d-%H%M%S") + '.csv')
