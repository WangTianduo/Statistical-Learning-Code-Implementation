import numpy as np
import time


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
	
	
