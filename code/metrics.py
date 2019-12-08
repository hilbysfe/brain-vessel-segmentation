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

def tversky_coef(y_true, y_pred, alpha, beta):
	"""Tverksy coefficient

	:param y_true: Ground truth target values.
	:param y_pred: Predicted targets returned by a model.
	"""
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	fp = K.sum(y_pred_f - (y_pred_f * y_true_f))
	fn = K.sum(abs(y_pred_f-1) - (abs(y_pred_f-1) * abs(y_true_f-1)))
	return intersection / ( intersection + alpha*fp + beta*fn )


def dice_coef_loss(y_true, y_pred):
	"""DICE loss function

	Computes the DICE loss function value.

	:param y_true: Ground truth target values.
	:param y_pred: Predicted targets returned by a model.
	:return: Negative value of DICE coefficient of the positive class in binary classification.
	"""
	return -dice_coef(y_true, y_pred, 1)


def weighted_dice_coef(y_true, y_pred, weights):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred * weights)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def weighted_dice_loss(weights):
	def weighted_dice(y_true, y_pred):
		return -weighted_dice_coef(y_true, y_pred, weights)
	return weighted_dice


def tversky_coef_loss(y_true, y_pred):
	"""DICE loss function

	Computes the DICE loss function value.

	:param y_true: Ground truth target values.
	:param y_pred: Predicted targets returned by a model.
	:return: Negative value of DICE coefficient of the positive class in binary classification.
	"""
	return -tversky_coef(y_true, y_pred, 0.4, 0.6)


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


def binary_conf_mat_values(y_true, y_pred):
	"""Binary confusion matrix

	Computes confusion matrix for a binary classification problem.

	:param y_true: Ground truth target values.
	:param y_pred: Predicted targets returned by a model.
	:return: True negatives. False positives. False negatives. True positives.
	"""
	y_true_f = y_true.flatten()
	y_pred_f = y_pred.flatten()
	conf_mat_values = confusion_matrix(y_true_f, y_pred_f).ravel().tolist()

	# the scikit-learn-confusion-matrix function returns only one number if all of the predicted targets are only
	# true negatives or true positives. The following code adds zeros to the other fields from the confusion matrix.
	if len(conf_mat_values) == 1:
		for i in range(3):
			conf_mat_values.append(0)
		# to check if the one number returned by the scikit-learn-confusion-matrix is value for true positives or
		# true negatives.
		if y_true.sum() / len(y_true_f) == 1:
			conf_mat_values = conf_mat_values[::-1]
	tn, fp, fn, tp = conf_mat_values
	return tn, fp, fn, tp
