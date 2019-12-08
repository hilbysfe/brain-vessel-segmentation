"""
This script contains a function that calculates performance measures such as AUC, accuracy, average class accuracy and
DICE coefficient for given predicted segmentation and ground truth label and saves the results to a given csv file.
"""

import csv
import pickle
import time
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from Unet import config
from Unet.utils import helper
from Unet.utils.metrics import avg_class_acc


def measure_performance_and_save_to_csv(patch_size, num_epochs, batch_size, lr, dropout, threshold, num_patients_train,
                                        num_patients_val, patients_segmentation, data_dirs, dataset, result_file):
    print('________________________________________________________________________________')
    print('patch size', patch_size)
    print('batch size', batch_size)
    print('learning rate', lr)
    print('dropout', dropout)
    print('threshold', threshold)

    start_row = time.time()

    # create the name of current run
    run_name = config.get_run_name(patch_size, num_epochs, batch_size, lr, dropout, num_patients_train,
                                   num_patients_val)
    print(run_name)

    # -----------------------------------------------------------
    # TRAINING RESULTS
    # -----------------------------------------------------------
    train_metadata_filepath = config.get_train_metadata_filepath(run_name)
    with open(train_metadata_filepath, 'rb') as handle:
        train_metadata = pickle.load(handle)

    print('Train params:')
    print(train_metadata['params'])
    print('Train performance:')
    tr_perf = train_metadata['performance']
    print(tr_perf)

    row = [patch_size, num_epochs, batch_size, lr, dropout, tr_perf['train_true_positives'],
           tr_perf['train_false_negatives'], tr_perf['train_false_positives'],
           tr_perf['train_true_negatives'], tr_perf['train_auc'], tr_perf['train_acc'],
           tr_perf['train_avg_acc'], tr_perf['train_dice'], tr_perf['val_true_positives'],
           tr_perf['val_false_negatives'], tr_perf['val_false_positives'],
           tr_perf['val_true_negatives'], tr_perf['val_auc'], tr_perf['val_acc'],
           tr_perf['val_avg_acc'], tr_perf['val_dice']]

    # -----------------------------------------------------------
    # VALIDATION / TEST RESULTS
    # -----------------------------------------------------------
    tp_list = []
    fn_list = []
    fp_list = []
    tn_list = []
    auc_list = []
    acc_list = []
    avg_acc_list = []
    dice_list = []

    for patient in patients_segmentation:
        print(patient)
        print('> Loading label...')
        label_mat = helper.load_nifti_mat_from_file(
            data_dirs[dataset] + patient + '_label.nii')  # values 0 or 1
        print('> Loading probability map...')
        prob_mat = helper.load_nifti_mat_from_file(
            config.get_probs_filepath(run_name, patient, dataset))  # values between 0 and 1
        pred_class = (prob_mat > threshold).astype(np.uint8)  # convert from boolean to int, values 0 or 1

        print()
        print('Computing performance measures...')
        label_mat_f = np.asarray(label_mat).flatten()
        prob_mat_f = np.asarray(prob_mat).flatten()
        pred_classes_f = np.asarray(pred_class).flatten()
        val_auc = roc_auc_score(label_mat_f, prob_mat_f)
        val_acc = accuracy_score(label_mat_f, pred_classes_f)
        val_avg_acc, val_tn, val_fp, val_fn, val_tp = avg_class_acc(label_mat_f, pred_classes_f)
        val_dice = f1_score(label_mat_f, pred_classes_f)

        tp_list.append(val_tp)
        fn_list.append(val_fn)
        fp_list.append(val_fp)
        tn_list.append(val_tn)
        auc_list.append(val_auc)
        acc_list.append(val_acc)
        avg_acc_list.append(val_avg_acc)
        dice_list.append(val_dice)

    row = row + [np.mean(tp_list), np.mean(fn_list), np.mean(fp_list), np.mean(tn_list), np.mean(auc_list),
                 np.mean(acc_list), np.mean(avg_acc_list), np.mean(dice_list)]
    print('Complete row:', row)

    print('Writing to csv...')
    with open(result_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    duration_row = int(time.time() - start_row)
    print('performance assessment took:', (duration_row // 3600) % 60, 'hours',
          (duration_row // 60) % 60, 'minutes',
          duration_row % 60,
          'seconds')
