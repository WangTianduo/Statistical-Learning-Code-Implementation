training = dict()

training['0'] = 1
training['1'] = 1
training['2'] = 1
training['3'] = -1
training['4'] = -1
training['5'] = -1
training['6'] = 1
training['7'] = 1
training['8'] = 1
training['9'] = -1

weight = [0.1 for _ in range(10)]

m = 3

def I(statement):
	if statement:
		return 1
	else:
		return -1
		
		
def get_error(train_data, weight, thres):
	total = len(train_data)
	wrong = 0
	for k, v in train_data.items():
		if I(int(k) < thres) * v < 0:
			wrong += 1
	return wrong / total
	
print(get_error(training, weight, 2.5))
	
def get_best_th(train_data, weight):
	return
	