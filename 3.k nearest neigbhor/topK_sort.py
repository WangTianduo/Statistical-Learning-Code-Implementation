import numpy as np

def heapify(arr, idx, length, maxheap=True):
	'''
	arr: the object of heapify
	idx: heapify which position
	length: the length of the arr
	maxheap: build max heap or min heap
	'''
	left = idx * 2 + 1
	right = idx * 2 + 2
	
	largest = idx
	
	if maxheap:
		if left < length and arr[left] > arr[idx]:
			largest = left
		
		if right < length and arr[right] > arr[largest]:
			largest = right
	else:
		if left < length and arr[left] < arr[idx]:
			largest = left
		
		if right < length and arr[right] < arr[largest]:
			largest = right
		
	if idx != largest:
		swap(arr, largest, idx)
		heapify(arr, largest, length, maxheap)
		
def swap(arr, a, b):
	temp = arr[a]
	arr[a] = arr[b]
	arr[b] = temp
	

def build_heap(arr, maxheap=True):
	length = len(arr)
	
	for i in range(int(length/2 - 1), -1, -1):
		heapify(arr, i, length, maxheap)
		
		
def set_top(arr, top, largest):
	arr[0] = top
	heapify(arr, 0, len(arr), not largest)
	
def topK(arr, k, largest):
	top = list()
	
	for i in range(k):
		top.append(arr[i])
		
	build_heap(top, not largest)
	for j in range(k, len(arr)):
		temp = top[0]
		
		if not largest:
			
			if arr[j] < temp:
				set_top(top, arr[j], largest)
		else:
			if arr[j] > temp:
				set_top(top, arr[j], largest)
			
	return top

import random

s = [x for x in range(20)]

random.shuffle(s)

print(topK(s, 5, False))
		