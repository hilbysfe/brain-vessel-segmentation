"""
This script uses the Evaluate_segmentation.exe from Taha for calculating performance measures.
"""

import time
import config
import os
from evaluate_segmentation_functions import evaluate_segmentation, parse_xml_to_csv, parse_xml_to_csv_avg_for_patients
from helper import read_tuned_params_from_csv
import numpy as np


class Evaluator():

		def __init__(self, patients, run_params, exec_path, eval_path, patients_dir, prob_dir):
		
			self.EXECUTABLE_PATH = exec_path
			self.patients = patients
			self.run_params = run_params
			self.EVAL_PATH = eval_path
			self.PATIENTS_DIR = patients_dir
			self.PROB_DIR = prob_dir

			return None


		
		# xml path for Evaluation tool
		def get_eval_segment_dataset_xmlpath(self, patient):
			return os.path.join(self.EVAL_PATH, 'eval_segment_' + patient + '.xml')
		# csv pathes for evaluation results
		def get_eval_segment_dataset_csvpath_per_patient(self):
			return os.path.join(self.EVAL_PATH, 'eval_segment_per_patient_' + time.strftime("%Y%m%d-%H%M%S") + '.csv')
		def get_eval_segment_dataset_csvpath(self):
			return os.path.join(self.EVAL_PATH, 'eval_segment_' + time.strftime("%Y%m%d-%H%M%S") + '.csv')

		def get_probs_filepath(self, patient):
			return os.path.join(self.PROB_DIR, 'probs_' + patient + '_.nii')
					   			 		  
		def evaluate_segmentations(self, threshold, measures):
			# create results folder for evaluation segmentation
			if not os.path.exists(self.EVAL_PATH):
				os.makedirs(self.EVAL_PATH)

			csv_path = self.get_eval_segment_dataset_csvpath()
			csv_path_per_patient = self.get_eval_segment_dataset_csvpath_per_patient()
			xml_paths = []

			for patient in self.patients:
				print('________________________________________________________________________________')
				print('patient:', patient)
			
				# load labels and segmentations
				label_path = os.path.join(self.PATIENTS_DIR, patient, config.LABEL_FILENAME)
				segmentation_path = self.get_probs_filepath(patient)

				# for saving results of evaluate segmentation to xml and to csv
				xml_path_patient = self.get_eval_segment_dataset_xmlpath(patient)
				xml_paths.append(xml_path_patient)

				# evaluation segmentation for patient
				evaluate_segmentation(label_path, segmentation_path, threshold, self.EXECUTABLE_PATH, xml_path_patient, measures)

				# parse the xml files in each folder, do stats and save the dataframes as csvs with the parse_xml
				# function
				run_params_patient = self.run_params
				run_params_patient['patient'] = patient
				parse_xml_to_csv(xml_path_patient, csv_path_per_patient, run_params_patient)

			parse_xml_to_csv_avg_for_patients(xml_paths, csv_path, self.run_params)

def main(model_def, dataset, patch_size, threshold, xval=False):
	################################################
	# SET PARAMETERS
	################################################
	data_dir = data_dir = config.ORIGINAL_DATA_DIR['all']
	num_epochs = config.NUM_EPOCHS  # number of epochs
	measures = "DICE,JACRD,AUC,KAPPA,RNDIND,ADJRIND,ICCORR,VOLSMTY,MUTINF,HDRFDST@0.95@,AVGDIST,MAHLNBS,VARINFO,GCOERR,PROBDST,SNSVTY,SPCFTY,PRCISON,FMEASR,ACURCY,FALLOUT,TP,FP,TN,FN,REFVOL,SEGVOL"
	batch_size = 64  # list with batch sizes
	learning_rate = 1e-4  # list with learning rates of the optimizer Adam
	dropout = 0.1 # percentage of weights to be dropped
	num_kernels = [32, 64, 128, 256]
	################################################

	executable_path = config.TOP_LEVEL + 'EvaluateSegmentation'
	   
	# PARAMETER LOOPS
	start_total = time.time()
	if xval:
		for fold in range(config.XVAL_FOLDS):

			patients = np.load(config.get_xval_fold_splits_filepath())[fold][dataset]
		
			run_params = run_params = {'patch size': patch_size, 'num epochs': num_epochs, 'batch size': batch_size,
								'learning rate': learning_rate, 'dropout': dropout, 'num_kernels': num_kernels}

			evaluator = Evaluator(
							patients=patients, 
							run_params=run_params, 
							exec_path=executable_path, 
							eval_path=os.path.join(config.RESULTS_DIR, str(fold), model_def, "eval_segment", dataset), 
							patients_dir=config.ORIGINAL_DATA_DIR[dataset], 
							prob_dir=os.path.join(config.RESULTS_DIR, str(fold), model_def, "probs", dataset))

			evaluator.evaluate_segmentations(threshold, measures)
	else:
					
		patients = os.listdir(config.ORIGINAL_DATA[dataset])
		run_params = run_params = {'patch size': patch_size, 'num epochs': num_epochs, 'batch size': batch_size,
							'learning rate': learning_rate, 'dropout': dropout, 'num_kernels': num_kernels}

		evaluator = Evaluator(
						patients=patients, 
						run_params=run_params, 
						exec_path=executable_path, 
						eval_path=os.path.join(config.RESULTS_DIR, model_def, "eval_segment", dataset), 
						patients_dir=config.ORIGINAL_DATA_DIR[dataset], 
						prob_dir=os.path.join(config.RESULTS_DIR, model_def, "probs", dataset))

		evaluator.evaluate_segmentations(threshold, measures)
		

	duration_total = int(time.time() - start_total)
	print('performance assessment took:', (duration_total // 3600) % 60, 'hours', (duration_total // 60) % 60, 'minutes',
		  duration_total % 60,
		  'seconds')
	print('DONE')


	
if __name__ == '__main__':
	main()
