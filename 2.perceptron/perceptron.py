from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import time

def perceptron(x, y, num_iter):
	
	x = np.mat(x)
	y = np.mat(y).T
	
	# m: number of examples; n: number of feature dimensions
	m, n = np.shape(x)
	
	# initialize the weights
	w = np.zeros((1, n))
	
	# initialize the bias
	b = 0
	
	# set default learning rate
	lr = 0.0001
	
	for k in range(num_iter):
		
		for i in range(m):
			xi = x[i]
			yi = y[i]
			
			if -1 * yi * (w * xi.T + b) >= 0:
				w = w + lr * yi * xi
				b = b + lr * yi
		print('Round %d:%d training' % (k, num_iter))
		
	return w, b

def test(x, y, w, b):
	x = np.mat(x)
	y = np.mat(y).T
	
	m, n = np.shape(x)
	
	errorCnt = 0
	
	for i in range(m):
		xi = x[i]
		yi = y[i]
		
		result = -1 * yi * (w * xi.T + b)
		
		if result >= 0: errorCnt += 1
		
	accuracy = 1 - (errorCnt / m)
	
	return round(accuracy, 2)
	
if __name__ == '__main__':
	
	data = load_breast_cancer(return_X_y=True)

	x_arr = data[0]
	y_arr = data[1]
	
	start = time.time()
	
	X_train, X_test, Y_train, Y_test = train_test_split(x_arr, y_arr, test_size=0.5)
	
	w, b = perceptron(x_arr, y_arr, 100)
	
	end = time.time()
	
	print('accuracy:{}'.format(test(X_test, Y_test, w, b)))
	print('time spent:{}'.format(round(end - start, 2)))
	
	