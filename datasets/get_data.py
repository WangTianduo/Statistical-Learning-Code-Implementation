from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class DataSetInfo:
	
	def __init__(self, raw_X, raw_Y):
		(sample_num, feature_num) = raw_X.shape
		self.x = raw_X
		self.y = raw_Y
		self.feature_num = feature_num
		self.sample_num = sample_num
		self.class_num = self.get_class_num()
		
	def split(self, test_ratio=0.3):
        # train_x, test_y, train_y, test_y
		return train_test_split(self.x, self.y, test_size=0.3)
		
	def get_class_num(self):
		temp = dict()
		
		for label in self.y:
			if str(label) not in temp:
				temp[str(label)] = 0
				
		return len(temp)


def get_dataset(name='breast_cancer'):
	
	if name == 'breast_cancer':
		(x, y) = load_breast_cancer(return_X_y=True)
		
		return DataSetInfo(x, y)
		
	else:
		print('Will be added soon!')
		