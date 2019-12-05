"""
This script uses the Evaluate_segmentation.exe from Taha for calculating performance measures.
"""

import time
from Full_vasculature.Utils import config
import os
from Full_vasculature.Utils.evaluate_segmentation_functions import evaluate_segmentation, parse_xml_to_csv, parse_xml_to_csv_avg_for_patients
from Unet.utils.helper import read_tuned_params_from_csv
import numpy as np


def main(model_def, xval=False):
	################################################
	# SET PARAMETERS
	################################################
	dataset = 'test'
	data_dir = data_dir = config.ORIGINAL_DATA_DIR['all']
	patch_size = config.PATCH_SIZES[model_def] 
	num_epochs = 10  # number of epochs
	threshold = config.THRESHOLD
	measures = "DICE,JACRD,AUC,KAPPA,RNDIND,ADJRIND,ICCORR,VOLSMTY,MUTINF,HDRFDST@0.95@,AVGDIST,MAHLNBS,VARINFO,GCOERR,PROBDST,SNSVTY,SPCFTY,PRCISON,FMEASR,ACURCY,FALLOUT,TP,FP,TN,FN,REFVOL,SEGVOL"
	################################################
	# GRID PARAMETERS
	################################################
	batch_size = 64  # list with batch sizes
	learning_rate = 1e-4  # list with learning rates of the optimizer Adam
	dropout = 0.1 # percentage of weights to be dropped
	num_kernels = [32, 64, 128, 256]
	################################################

	executable_path = config.EXECUTABLE_PATH
	   
	# PARAMETER LOOPS
	start_total = time.time()
	if xval:
		for fold in range(config.XVAL_FOLDS):
			# create results folder for evaluation segmentation
			if not os.path.exists(config.get_eval_segment_dataset_path(model_def, dataset, fold)):
				os.makedirs(config.get_eval_segment_dataset_path(model_def, dataset, fold))
			
			patients = np.load(config.get_xval_fold_splits_filepath())[fold][dataset]
		
			csv_path = config.get_eval_segment_dataset_csvpath(model_def, dataset, fold)
			csv_path_per_patient = config.get_eval_segment_dataset_csvpath_per_patient(model_def, dataset, fold)
			xml_paths = []
			for patient in patients:
				print('________________________________________________________________________________')
				print('patient:', patient)
			
				# load labels and segmentations
				label_path = os.path.join(data_dir, patient, config.LABEL_FILENAME)
				segmentation_path = config.get_probs_filepath(patient, model_def, dataset, fold)
				# for saving results of evaluate segmentation to xml and to csv
				xml_path_patient = config.get_eval_segment_dataset_xmlpath(patient, model_def, dataset, fold)
				xml_paths.append(xml_path_patient)

				# evaluation segmentation for patient
				evaluate_segmentation(label_path, segmentation_path, threshold, executable_path, xml_path_patient, measures)

				# parse the xml files in each folder, do stats and save the dataframes as csvs with the parse_xml
				# function
				run_params = {'patch size': patch_size, 'num epochs': num_epochs, 'batch size': batch_size,
								'learning rate': learning_rate, 'dropout': dropout, 'num_kernels': num_kernels, 'patient': patient}
				parse_xml_to_csv(xml_path_patient, csv_path_per_patient, run_params)

			run_params = {'patch size': patch_size, 'num epochs': num_epochs, 'batch size': batch_size,
							'learning rate': learning_rate, 'dropout': dropout, 'num_kernels': num_kernels}
			parse_xml_to_csv_avg_for_patients(xml_paths, csv_path, run_params)
	else:
		# create results folder for evaluation segmentation
		if not os.path.exists(config.get_eval_segment_dataset_path(model_def, dataset)):
			os.makedirs(config.get_eval_segment_dataset_path(model_def, dataset))
			
		patients = os.listdir(config.ORIGINAL_DATA[dataset])
		
		csv_path = config.get_eval_segment_dataset_csvpath(model_def, dataset)
		csv_path_per_patient = config.get_eval_segment_dataset_csvpath_per_patient(model_def, dataset)
		xml_paths = []
		for patient in patients:
			print('________________________________________________________________________________')
			print('patient:', patient)
			
			# load labels and segmentations
			label_path = os.path.join(data_dir, patient, config.LABEL_FILENAME)
			segmentation_path = config.get_probs_filepath(patient, model_def, dataset)
			# for saving results of evaluate segmentation to xml and to csv
			xml_path_patient = config.get_eval_segment_dataset_xmlpath(patient, model_def, dataset)
			xml_paths.append(xml_path_patient)

			# evaluation segmentation for patient
			evaluate_segmentation(label_path, segmentation_path, threshold, executable_path, xml_path_patient, measures)

			# parse the xml files in each folder, do stats and save the dataframes as csvs with the parse_xml
			# function
			run_params = {'patch size': patch_size, 'num epochs': num_epochs, 'batch size': batch_size,
							'learning rate': learning_rate, 'dropout': dropout, 'num_kernels': num_kernels, 'patient': patient}
			parse_xml_to_csv(xml_path_patient, csv_path_per_patient, run_params)

		run_params = {'patch size': patch_size, 'num epochs': num_epochs, 'batch size': batch_size,
						'learning rate': learning_rate, 'dropout': dropout, 'num_kernels': num_kernels}
		parse_xml_to_csv_avg_for_patients(xml_paths, csv_path, run_params)

	duration_total = int(time.time() - start_total)
	print('performance assessment took:', (duration_total // 3600) % 60, 'hours', (duration_total // 60) % 60, 'minutes',
		  duration_total % 60,
		  'seconds')
	print('DONE')


	
if __name__ == '__main__':
	main()
