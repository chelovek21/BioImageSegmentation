"""
File contains methods for:
    * getting most representative subset
    * different acquisistion functions

Methods to use:
max_representativeness(S_u, S_c, Sc_idx, small_k)
acquisition_max_variance(Mc_pred)
acquisition_max_entropy(Mc_pred)
"""

from __future__ import division
import numpy as np
from skimage.transform import rotate

__author__ = "Mathias Baltzersen and Rasmus Hvingelby"


###############################
# Data augmentation for batch #
###############################

def augment_batch(x_train, y_train_seg, y_train_cont, size):
    x_train_aug = []
    y_train_seg_aug = []
    y_train_cont_aug = []

    for x_img, y_seg_img, y_cont_img in zip(x_train, y_train_seg, y_train_cont):

        h = x_img.shape[0]
        w = x_img.shape[1]

        x1 = np.random.randint(0, w - size)
        y1 = np.random.randint(0, h - size)

        x2 = x1 + size
        y2 = y1 + size

        tmp_x = x_img[y1:y2, x1:x2, :]
        tmp_y_seg = y_seg_img[y1:y2, x1:x2, :]
        tmp_y_cont = y_cont_img[y1:y2, x1:x2, :]

        # Rotation
        angel = np.random.randint(0, 4) * 90

        tmp_x = np.array(rotate(tmp_x, angel, preserve_range=True), dtype=np.uint8)
        tmp_y_seg = np.array(rotate(tmp_y_seg, angel, preserve_range=True), dtype=np.uint8)
        tmp_y_cont = np.array(rotate(tmp_y_cont, angel, preserve_range=True), dtype=np.uint8)

        if np.random.rand() > 0.5:
            tmp_x = tmp_x[::-1, :, :]
            tmp_y_seg = tmp_y_seg[::-1, :, :]
            tmp_y_cont = tmp_y_cont[::-1, :, :]

        x_train_aug.append(tmp_x)
        y_train_seg_aug.append(tmp_y_seg)
        y_train_cont_aug.append(tmp_y_cont)

    return np.array(x_train_aug), np.array(y_train_seg_aug), np.array(y_train_cont_aug)


####################################
# Functions for representativeness #
####################################
def max_representativeness(S_u, S_c, Sc_idx, small_k):
    print("Finding most representative subset")

    S_a_idx = []
    S_a = []

    while len(S_a) < small_k or not S_c:  # last iteration we might not have k images to choose from
        current_best = 0
        current_best_idx = None
        for i, img_and_idx in enumerate(zip(S_c, Sc_idx)):
            S_a.append(img_and_idx[0])
            tmp_score = _big_f(S_a, S_u)
            if tmp_score > current_best:
                current_best = tmp_score
                current_best_idx = i

            S_a.pop()

        S_a.append(S_c[current_best_idx])
        S_a_idx.append(Sc_idx[current_best_idx])
        S_c.pop(current_best_idx)
        Sc_idx.pop(current_best_idx)

    return S_a_idx


def _small_f(S_a, I_x):
    # return the sim of the image in S_a with highest sim with I_x
    max_sim = 0
    max_sim_idx = 0
    for I_sa in S_a:
        sim = _cos_sim(I_sa, I_x)
        if sim > max_sim:
            max_sim = sim

    return max_sim


def _big_f(S_a, S_u):
    # Sum small_f for all images in S_u f_small(S_a, I_from_S_u)
    current_sum = 0
    for I_su in S_u:
        current_sum += _small_f(S_a, I_su)

    return current_sum


def _cos_sim(I_i, I_j):
    return np.dot(I_i, I_j.T) / I_i.shape[0] ** 2


###################################
# Different acquisition functions #
###################################


def acquisition_max_variance(Mc_pred):
    """
    Used in 'suggestive annotation a deep active learning framework for biomedical image segmentation'
    :param Mc_pred: Mc_pred: ndarray of shape [num_mc_samples, pool_size, img_h, img_w, classes]
    :param classmethod is a string which selects a method for handling the class dimension of the data.
    Valid strings are:  'mean', 'entropy', 'least_confident', 'margin'
    :return: [pool_size]
    """

    Mc_pred = np.var(Mc_pred, axis=0, keepdims=False)  # Variance amoung the models/committee
    Mc_pred = np.mean(Mc_pred, axis=(1, 2), keepdims=False)  # Mean over hight, width

    pool = np.sum(Mc_pred, axis=1, keepdims=False)  # Sum class variances

    return pool


def acquisition_max_entropy(Mc_pred):
    """
    Also called Soft vote entropy.

    :param Mc_pred: ndarray of shape [num_mc_samples, pool_size, img_h, img_w, classes]
    :param classmethod is a string which selects a method for handling the class dimension of the data.
    Valid strings are:  'mean', 'entropy', 'least_confident', 'margin'
    :return: max entropy with shape[pool_size]
    """

    Mc_pred = np.mean(Mc_pred, axis=(0, 2, 3), keepdims=False)  # Mean over mc_samples, height and width
    entropy = -np.sum(np.multiply(Mc_pred, np.log10(Mc_pred)), axis=1)  # Sum over classes

    return entropy


def acquisition_KL_divergence(Mc_pred):
    P_c = np.mean(Mc_pred, axis=(0, 2, 3), keepdims=False)  # Mean over mc_samples, height and width
    P_theta = np.mean(Mc_pred, axis=(2, 3), keepdims=False)  # [num_mc_samples, pool_size, classes]
    KL = np.sum(P_theta * np.log10((P_theta / P_c)), axis=2)  # [num_mc_samples, pool_size]

    committee_KL = np.mean(KL, axis=0)  # [pool_size]

    return committee_KL


####################################################
# Functions to handle the class dimension.         #
# They all take in class_prob=[pool_size, classes] #
####################################################
'''
OBESLETE for the time being

def _mean(class_probs):
    return np.mean(class_probs, axis=1, keepdims=False)


def _entropy(class_probs):
    return -np.sum(np.multiply(class_probs, np.log10(class_probs)), axis=1)


def _least_confident(class_probs):
    return 1 - np.max(class_probs, axis=1, keepdims=False)


def _margin(class_probs):
    y_top2 = np.sort(class_probs, axis=1)[:, -2:]
    return y_top2[:, 1] - y_top2[:, 0]

'''


def acquisition_func(method, mc_pred):
    switcher = {
        'KL_divergence': acquisition_KL_divergence,
        'variance': acquisition_max_variance,
        'entropy': acquisition_max_entropy,
    }

    func = switcher.get(method)

    if func is None:
        raise Exception("Invalid class method chosen. Valids are:'variance', 'entropy' and 'KL_divergence'")

    return func(mc_pred)

def should_evaluate(evaluate_intervals, percentage_data_used):
    """
    This is the worst method I have ever written!
    The basic idea is that this method will see
    if the percentage of data used for training
    is larger than any of the intervals we
    want to evaluate our model at. If it
    is then we "should_evaluate" and
    we remove the interval from the
    evaluate_intervals list.

    Feel free to rewrite or anything!

    Author: Rasmus Hvingelby

    :param evaluate_intervals:
    :param percentage_data_used:
    :return:
    """
    updated_evaluate_intervals = []

    for evaluate_interval in evaluate_intervals:
        if percentage_data_used <= evaluate_interval:
            updated_evaluate_intervals.append(evaluate_interval)

    should_eval = len(evaluate_intervals) > len(updated_evaluate_intervals)

    return updated_evaluate_intervals, should_eval