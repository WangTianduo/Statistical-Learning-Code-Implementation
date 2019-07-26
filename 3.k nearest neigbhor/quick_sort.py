def quick_sort_1(array, l, r):
	if l < r:
		q = partition(array, l ,r)
		quick_sort_1(array, l, q - 1)
		quick_sort_1(array, q + 1, r)
		 

def partition(array, l, r):
	x = array[r]
	i = l - 1
	for j in range(l, r):
		if array[j] <= x:
			i += 1
			array[i], array[j] = array[j], array[i]
			
	array[i+1], array[r] = array[r], array[i+1]
	
	return i + 1
	
array = [x for x in range(20, 0, -1)]

print(array)

	
def quick_sort_2(array, left, right):
	if left >= right:
		return
	
	low = left
	high = right
	key = array[low]
	
	while left < right:
		while left < right and array[right] > key:
			right -= 1
		array[left] = array[right]
		while left < right and array[left] <= key:
			left += 1
		array[right] = array[left]
		
	array[right] = key
	quick_sort_2(array, low, left-1)
	quick_sort_2(array, left + 1, high)
	
quick_sort_2(array, 0, 19)

print(array)