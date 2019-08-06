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
    D_x = np.array(D_x)
    D_y = np.array(D_y).T

    discrete_x = D_x

    feature_num = D_x.shape[1]

    max_G_D_A = -1
    max_feature = -1

    H_D = cal_shanno_ent(D_y)
    # print(H_D)
    for feature in range(feature_num):

        temp = np.array(discrete_x[:, feature].flat)
        G_D_A = H_D - cal_H_D_A(temp, D_y)
        # print(G_D_A, feature)
        if G_D_A > max_G_D_A:
            max_G_D_A = G_D_A
            max_feature = feature
    return max_feature, max_G_D_A


def majorClass(labelArr):
    '''
    找到当前标签集中占数目最大的标签
    :param labelArr: 标签集
    :return: 最大的标签
    '''
    #建立字典，用于不同类别的标签技术
    classDict = {}
    #遍历所有标签
    for i in range(len(labelArr)):
        #当第一次遇到A标签时，字典内还没有A标签，这时候直接幅值加1是错误的，
        #所以需要判断字典中是否有该键，没有则创建，有就直接自增
        if labelArr[i] in classDict.keys():
            # 若在字典中存在该标签，则直接加1
            classDict[labelArr[i]] += 1
        else:
            #若无该标签，设初值为1，表示出现了1次了
            classDict[labelArr[i]] = 1
    #对字典依据值进行降序排序
    classSort = sorted(classDict.items(), key=lambda x: x[1], reverse=True)
    #返回最大一项的标签，即占数目最多的标签
    return classSort[0][0]


def get_subdata_array(train_x, train_y, A, a):
    '''
    update dataset
    :param train_x, train_y: dataset
    :param A: the index of the eliminting feature
    :param a: if data[A] == a, then keep the line
    :return: new dataset
    '''
    retDataArr = []
    retLabelArr = []

    for i in range(len(train_x)):
        if train_x[i][A] == a:
            retDataArr.append(np.concatenate([train_x[i][0:A],train_x[i][A+1:]]))
            retLabelArr.append(train_y[i])

    return retDataArr, retLabelArr


def create_tree(*dataSet):
    '''
    recursively create decision tree
    :param dataSet: (train_x, train_y)
    :return: ??
    '''

    Epsilon = 0.1

    train_x = dataSet[0][0]
    train_y = dataSet[0][1]

    # print('train_x shape:{}'.format(train_x.shape))
    # print('train_y shape:{}'.format(train_y.shape))
    print('start a node: ', len(train_x[0]), len(train_y))

    classDict = {i for i in train_y}

    if len(classDict) == 0:
        return majorClass(train_y)
    elif len(classDict) == 1:
        return train_y[0]
    else:
        Ag, EpsilonGet = get_best_feature(train_x, train_y)

        if EpsilonGet < Epsilon:
            return majorClass(train_y)

        treeDict = {Ag:{}}

        treeDict[Ag][0] = create_tree(get_subdata_array(train_x, train_y, Ag, 0))
        treeDict[Ag][1] = create_tree(get_subdata_array(train_x, train_y, Ag, 1))

        return treeDict


def predict(test_x, tree):
    # test_x = test_x.tolist()
    while True:
        (key, value), = tree.items()

        if type(tree[key]).__name__ == 'dict':
            dataVal = test_x[key]

            print(test_x)
            test_x = np.concatenate([test_x[0:key],test_x[key+1:]])

            tree = value[dataVal]
            print(tree)

            if type(tree).__name__ != 'dict':
                return tree
        else:
            return value

def test(test_x, test_y, tree):
    
    error_cnt = 0
    for i in range(len(test_x)):

        if test_y[i] != predict(test_x[i], tree):
            error_cnt += 1

    return 1 - error_cnt / len(test_x)


if __name__ == '__main__':
    start = time.time()

    cancer_set_info = get_dataset('breast_cancer')
    train_x, test_x, train_y, test_y = cancer_set_info.split()

    train_x = discretize(train_x, 2)
    tree = create_tree((train_x, train_y))

    print('tree:{}'.format(tree))
    test_x = discretize(test_x, 2)
    acc = test(test_x, test_y, tree)
    print('accuracy is {}'.format(acc))

    end = time.time()
    print('time spent:{}'.format(end-start))


