"""
This file defines the metrics and loss functions used for network training and performance assessment.
"""

from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix


def dice_coef(y_true, y_pred, smooth=0):
	"""DICE coefficient

	Computes the DICE coefficient, also known as F1-score or F-measure.

	:param y_true: Ground truth target values.
	:param y_pred: Predicted targets returned by a model.
	:param smooth: Smoothing factor.
	:return: DICE coefficient of the positive class in binary classification.
	"""
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
	"""DICE loss function

	Computes the DICE loss function value.

	:param y_true: Ground truth target values.
	:param y_pred: Predicted targets returned by a model.
	:return: Negative value of DICE coefficient of the positive class in binary classification.
	"""
	return -dice_coef(y_true, y_pred, 1)


def avg_class_acc(y_true, y_pred):
	"""Average class accuracy

	:param y_true: Ground truth target values.
	:param y_pred: Predicted targets returned by a model.
	:return: Average class accuracy. True negatives. False positives. False negatives. True positives.
	"""
	tn, fp, fn, tp = binary_conf_mat_values(y_true, y_pred)
	P_acc = tp / (tp + fn)
	N_acc = tn / (tn + fp)
	avg_acc = (P_acc + N_acc) / 2
	return avg_acc, tn, fp, fn, tp