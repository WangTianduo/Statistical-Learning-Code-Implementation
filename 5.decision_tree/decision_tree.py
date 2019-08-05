import numpy as np
import time

from math import log

import sys
sys.path.append('..')

from datasets.get_data import get_dataset
from utils import discretize

def cal_shanno_ent(train_y):
    num_entries = len(train_y)
    label_count = dict()

    for idx in range(len(train_y)):
        if train_y[idx] not in label_count:
            label_count[train_y[idx]] = 0
        label_count[train_y[idx]] += 1

    entropy = 0

    for key in label_count:
        prob = float(label_count[key] / num_entries)
        entropy -= prob * log(prob, 2)
    
    return entropy


# H: entropy; D: training set; A: feature A
def cal_H_D_A(D_x_A, D_y):
    '''
    calculate empirical entropy
    :param D_x_A: train set X with only feature A (discrete)
    :param D_y: train set Y
    :return empirical entropy of A
    '''
    H_D_A = 0
    train_X_set = set([label for label in D_x_A])
    train_Y_set = set([label for label in D_y])

    D = len(D_x_A)
    for i in train_X_set:
        Di = 0
        for label in D_x_A:
            if label == i:
                Di += 1
        ratio = Di / D
        for k in train_Y_set:
            Dik = 0
            for idx in range(D):
                if D_x_A[idx] == i and D_y[idx] == k:
                    Dik += 1
            prob = Dik / Di
            if Dik != 0:
                H_D_A -= ratio * prob * log(prob, 2)
    return H_D_A


def get_best_feature(D_x, D_y):
    '''
    get the feature with higest information gain
    :param D_x: training X set
    :param D_y: training Y set
    return feature idx
    '''

    # discrete_x = discretize(D_x, 10)
    discrete_x = D_x

    feature_num = D_x.shape[1]

    max_G_D_A = -1
    max_feature = -1

    H_D = cal_shanno_ent(D_y)
    print(H_D)
    for feature in range(feature_num):

        temp = np.array(discrete_x[:, feature].flat)
        G_D_A = H_D - cal_H_D_A(temp, D_y)
        print(G_D_A, feature)
        if G_D_A > max_G_D_A:
            max_G_D_A = G_D_A
            max_feature = feature
    return max_feature, max_G_D_A


cancer_set_info = get_dataset('breast_cancer')

X_train, X_test, Y_train, Y_test = cancer_set_info.split()

print(get_best_feature(X_train,  Y_train))


