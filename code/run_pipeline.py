

import train_unet
import predict_full_brain
import evaluate_segmentation
import compute_error_masks
import patch_extraction_3d



class Vessel_segmentation():

	def __init__(self):

		self.TOP_LEVEL = r"/home/adam/code/Vessel_segmentation/"

		DATA_PATH = r"C:\Users\Adam Hilbert\Data_M2\FV_data"
		MODEL_PATH = r"C:\Users\Adam Hilbert\Data_M2\FV_data\models\\"
		MODEL_DATA_PATH = r"C:\Users\Adam Hilbert\Data_M2\FV_data\model_data"
		RESULTS_DIR = r"/data-raid5/adam/result"

		# ----------------
		# PATCH SETTINGS 
		# ----------------
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
		
		NUM_PATCHES = 2000  # number of patches we want to extract from one stack (one patient)

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

		ORIGINAL_DATA_DIR = {'all': os.path.join(DATA_PATH, 'original_data', 'all')}
		#					 'train': os.path.join(DATA_PATH, 'original_data', 'train'),
		#					 'test': os.path.join(DATA_PATH, 'original_data', 'test'),
		#					 'val': os.path.join(DATA_PATH, 'original_data', 'val')}

		# original files with scans
		IMG_FILENAME = '001.nii'
		LABEL_FILENAME = '001_Vessel-Manual-Gold-int.nii'

		XVAL_FOLDS = 4

		return None

	# Splits of patients stored in dictionary
	def get_xval_fold_splits_filepath():
		return os.path.join(MODEL_DATA_PATH, "xval_folds.npy")

	


	



def main():

	# --- Run training + full brain prediction + evaluation
	models = [
#		"unet-3d",
#		"context-unet-3d",
#		"ds-unet-3d",
		"brainseg-2d"
	]

	for model_def in models:
		print("Model: " + model_def)
		train_unet.main(model_def)
		predict_full_brain.main(model_def)
		evaluate_segmentation.main(model_def)
		compute_error_masks.main(model_def)

if __name__ == '__main__':
	main()