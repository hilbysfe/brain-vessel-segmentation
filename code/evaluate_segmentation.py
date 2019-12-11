"""
This script uses the Evaluate_segmentation.exe from Taha for calculating performance measures.
"""

import time
import os
from evaluate_segmentation_functions import evaluate_segmentation, parse_xml_to_csv, parse_xml_to_csv_avg_for_patients
from helper import read_tuned_params_from_csv
import numpy as np


class Evaluator():

		def __init__(self, patients, run_params, exec_path, eval_path, label_files, prob_files):
		
			self.EXECUTABLE_PATH = exec_path
			self.patients = patients
			self.run_params = run_params
			self.EVAL_PATH = eval_path
			self.label_files = label_files
			self.prob_files = prob_files

			return None
		
		# xml path for Evaluation tool
		def get_eval_segment_dataset_xmlpath(self, patient):
			return os.path.join(self.EVAL_PATH, 'eval_segment_' + patient + '.xml')
		# csv pathes for evaluation results
		def get_eval_segment_dataset_csvpath_per_patient(self):
			return os.path.join(self.EVAL_PATH, 'eval_segment_per_patient_' + time.strftime("%Y%m%d-%H%M%S") + '.csv')
		def get_eval_segment_dataset_csvpath(self):
			return os.path.join(self.EVAL_PATH, 'eval_segment_' + time.strftime("%Y%m%d-%H%M%S") + '.csv')

					   			 		  
		def evaluate_segmentations(self, threshold, measures):
			
			start_total = time.time()
			# create results folder for evaluation segmentation
			if not os.path.exists(self.EVAL_PATH):
				os.makedirs(self.EVAL_PATH)

			csv_path = self.get_eval_segment_dataset_csvpath()
			csv_path_per_patient = self.get_eval_segment_dataset_csvpath_per_patient()
			xml_paths = []

			for i, patient in enumerate(self.patients):
				print('________________________________________________________________________________')
				print('patient:', patient)
			
				# load labels and segmentations
				label_path = self.label_files[i]
				segmentation_path = self.prob_files[i]

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

			duration_total = int(time.time() - start_total)
			print('performance assessment took:', (duration_total // 3600) % 60, 'hours', (duration_total // 60) % 60, 'minutes',
				  duration_total % 60,
				  'seconds')

			return

