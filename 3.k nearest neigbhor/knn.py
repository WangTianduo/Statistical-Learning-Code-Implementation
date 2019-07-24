from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import numpy as np
import time

def getDist(x1, x2):
	'''
	x1: numpy array
	x2: numpy array
	'''
	
	return np.sqrt(np.sum(np.square(x1 - x2)))
	
	
def getCloset(x_arr, y_arr, target, topK):
	
	'''
	x_arr: training set: feature
	y_arr: training set: label
	target: sample that will be labeled
	topK: K-NN
	'''
	
	distList = [0] * len(x_arr)
	
	for i in range(len(x_arr)):
		xi = x_arr[i]
		
		distance = getDist(xi, target)
		distList[i] = distance
		
	topK_list = np.argsort(np.array(distList))[:topK]
	
	# here 2 is the number of classes in the dataset
	labelList = [0] * 2
	
	for idx in topK_list:
		labelList[int(y_arr[idx])] += 1
		
	return labelList.index(max(labelList))
	
def test(x_train, y_train, x_test, y_test, topK):
	
	x_train = np.mat(x_train)
	x_test = np.mat(x_test)
#	y_train = np.mat(y_train)
#	y_test = np.mat(y_test)
	
	errorCnt = 0
	
	for i in range(len(x_test)):
		print('test %d: %d'%(i, len(x_test)))
		
		x = x_test[i]
		
		pred_y = getCloset(x_train, y_train, x, topK)
		
		if pred_y != y_test[i]: errorCnt += 1
		
	return 1 - (errorCnt / len(x_test))
	
if __name__ == '__main__':
	
	data = load_breast_cancer(return_X_y=True)

	x_arr = data[0]
	y_arr = data[1]
		
	start = time.time()
		
	X_train, X_test, Y_train, Y_test = train_test_split(x_arr, y_arr, test_size=0.5)
	
	accuracy = test(X_train, Y_train, X_test, Y_test, 8)
	
	end = time.time()
	print('accuracy is: {}%'.format(accuracy * 100))
	print('time spent: {}'.format(round(end - start, 2)))
