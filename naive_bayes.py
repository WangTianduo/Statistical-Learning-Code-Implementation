import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def divide_group(min_value, max_value, x, gap_num):
	'''
	Function that discretize continuous variable
	'''

	gap_length = (max_value - min_value+1) / gap_num
	return int(np.floor((x - min_value)/gap_length))
	

def discretize(origin_x, gap_num):
	
	'''
	:param origin_x: continuous R.V.; assume 2D matrix
	:param gap_num: number of slots 
	'''
	
	# transpose the original x matrix
	origin_x = origin_x.T
	(feature_num, sample_num) = origin_x.shape
	
	new_x = np.zeros((feature_num, sample_num), dtype=int)
	
	for i in range(feature_num):
		min_value = min(origin_x[i])
		max_value = max(origin_x[i])
		
		for j in range(sample_num):
			new_x[i][j] = divide_group(min_value, max_value, origin_x[i][j], gap_num)
	
	return new_x.T
	

def get_distribution(train_x, train_y):
	
	'''
	:param train_x: after discretization
	:param train_y: original one
	'''
	feature_num = 30
	class_num = 2
	feature_value_split = 20
	p_y = np.zeros((class_num, 1))
	
	for i in range(class_num):
		p_y[i] = ((np.sum(np.mat(train_y))==i) + 1) / (len(train_y)+class_num)
		
	p_y = np.log(p_y)
		
	p_xy = np.zeros((class_num, feature_num, feature_value_split))
	
	for i in range(len(train_y)):
		label = train_y[i]
		
		x = train_x[i]
		for j in range(feature_num):
			p_xy[label][j][x[j]] += 1	
			
	for label in range(class_num):
		for j in range(feature_num):
			temp_sum = np.sum(p_xy[label][j])
			for k in range(feature_value_split):
				p_xy[label][j][k] = np.log((p_xy[label][j][k] + 1) / (temp_sum + feature_value_split))
				
	return p_y, p_xy
		

def naive_bayes(p_y, p_xy, x):
	'''
	:param p_y: prior probability distribution
	:param p_xy: conditional distribution
	:param x: example to be predicted
	:return: predicted label of x
	'''
	
	feature_num = 30
	class_num = 2
	
	P = [0] * class_num
	
	for i in range(class_num):
		
		sum = 0
		
		for j in range(feature_num):
			sum += p_xy[i][j][x[j]]
			
		P[i] = sum + p_y[i]
		
	return P.index(max(P))
	

def test(p_y, p_xy, test_x, test_y):
	
	errorCnt = 0
	for i in range(len(test_x)):
		
		predict_y = naive_bayes(p_y, p_xy, test_x[i])
		
		if predict_y != test_y[i]:
			errorCnt += 1
			
	return 1 - errorCnt / len(test_x)
	

if __name__ == '__main__':
	start = time.time()
	
	(x_arr, y_arr) = load_breast_cancer(return_X_y=True)
	X_train, X_test, Y_train, Y_test = train_test_split(x_arr, y_arr, test_size=0.3)
	
	discret_train_X = discretize(X_train, 20)
	p_y, p_xy = get_distribution(discret_train_X, Y_train)
	
	discret_test_X = discretize(X_test, 20)
	accuracy = test(p_y, p_xy, discret_test_X, Y_test)
	
	end = time.time()
	
	print('accuracy:{}'.format(accuracy))
	print('time_spent:{}s'.format(end - start))
