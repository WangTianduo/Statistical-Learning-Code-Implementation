import numpy as np

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