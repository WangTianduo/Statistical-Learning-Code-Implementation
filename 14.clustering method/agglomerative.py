import numpy as np

# 李航：统计学习方法（第二版）
# 14.2 层次聚类 例 14.1 Pg 262 


distance = np.array([[0, 7, 2, 9, 3],
					[7, 0, 5, 4, 6],
					[2, 5, 0, 8, 1],
					[9, 4, 8, 0, 5],
					[3, 6, 1, 5, 0]])

min_distance = distance[1][0]
min_pair = (0, 0)
for i in range(distance.shape[0]):
	for j in range(distance.shape[1]):
		if distance[i][j] < min_distance and i!=j:
			min_distance = distance[i][j]
			min_pair = (i, j)
			
def agglo_cluster(D):
	jump_idx = []
	cluster_map = dict()
	
	original_max = len(D)
	for i in range(original_max):
		cluster_map[str(i)] = i
	
	while len(jump_idx) < D.shape[0]-1:
		_, min_pair = getmin(D, jump_idx)
		cluster_map[str(original_max)] = min_pair
		original_max += 1
		jump_idx.extend(min_pair)
		D = add_new_cluster(D, min_pair, jump_idx)
		
	return cluster_map
	
def getmin(D, jump_idx):
	# assume D is equal or larger than 2*2
	# assume jump_idx is a list
	min_distance = distance[1][0]
	min_pair = (1, 0)
	for i in range(D.shape[0]):
		if i in jump_idx:
			continue
		else:
			for j in range(D.shape[1]):
				if j in jump_idx:
					continue
				else:
					if D[i][j] < min_distance and i!=j:
						min_distance = D[i][j]
						min_pair = (i, j)
	return min_distance, min_pair
	
def add_new_cluster(D, pair, jump_idx):
	new_d = np.zeros((D.shape[0]+1, D.shape[1]+1))
	for i in range(D.shape[0]):
		for j in range(D.shape[1]):
			new_d[i][j] = D[i][j]
	
	for i in range(new_d.shape[0]-1):
		if i in pair:
			continue
		else:
			new_d[i][new_d.shape[0]-1] = min([D[i][x] for x in range(D.shape[0]) if x!=i and x in pair])
			new_d[new_d.shape[0]-1][i] = min([D[i][x] for x in range(D.shape[0]) if x!=i and x in pair])
	return new_d
	
if __name__ == '__main__':
	print(agglo_cluster(distance))