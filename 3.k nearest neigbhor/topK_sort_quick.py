
def partition(array, low, high):
	if low < high:
		flag = array[low]
		
		while low < high:
			while low < high and array[high] >= flag:
				high -= 1
			
			array[low] = array[high]
			
			while low < high and array[low] < flag:
				low += 1
			
			array[high] = array[low]
		
		array[low] = flag
		return low
		
	else:
		return 0
		

def get_topK(array, k):
	
	low = 0
	high = len(array) - 1
	index = partition(array, low, high)
	
	while index != k - 1:
		if index > k - 1:
			high = index - 1
			index = partition(array, low, high)
			
		if index < k - 1:
			low = index + 1
			index = partition(array, low, high)
			
	return array[-k:]
			
s = [x for x in range(20, 1, -1)]

print(get_topK(s, 5))