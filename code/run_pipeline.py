

from Full_vasculature.Training import train_unet
from Full_vasculature.Unet import predict_full_brain
from Full_vasculature.Evaluation import evaluate_segmentation
from Full_vasculature.Utils import compute_error_masks
from Full_vasculature.data_processing import patch_extraction_3d



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
#		train_unet.main(model_def)
		predict_full_brain.main(model_def)
		evaluate_segmentation.main(model_def)
		compute_error_masks.main(model_def)

if __name__ == '__main__':
	main()