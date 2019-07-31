import numpy as np
import time

from math import log

import sys
sys.path.append('..')

from datasets.get_data import get_dataset

def cal_shanno_ent(train_x, train_y):
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

cancer_set_info = get_dataset('breast_cancer')

train_x, test_x, train_y, test_y = cancer_set_info.split()

print(cal_shanno_ent(train_x, train_y))


