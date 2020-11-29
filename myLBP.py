import numpy as np
import math
import cv2
import time
from matplotlib import pyplot as plt
import statistics

class customMe:                             #Main LBP Class. Initialized with block size 
	def __init__(self, block_size):
		self.block_size=block_size

	def findLBP(self,img):                   #Main function to calculate LBP. Input: Image, Output : LBP Codes
		(N, M) = img.shape
		codes=[]
		r = 0
		while True:
			c = 0
			while True:                                             #Loop Selects a 3x3 window over which LBP is to be computed.
				sliceMe = img[r:r + 3, c:c + 3]
				thresh = sliceMe[1, 1]
				sliceCopy = sliceMe
				sliceCopy = np.where(sliceCopy >= thresh, 1, 0)     #Thresholding with center element 
				code = sliceCopy[0, 0] * 128 + sliceCopy[0, 1] * 64 + sliceCopy[0, 2] * 32 + sliceCopy[1, 2] * 16 + \
				       sliceCopy[2, 2] * 8 + sliceCopy[2, 1] * 4 + sliceCopy[2, 0] * 2 + sliceCopy[1, 0]
				codes.append(code)                                    #Appending the code calculated into a list 
				c = c + 3
				if c + 2 > M - 1:
					break
			r = r + 3
			if r + 2 > N - 1:
				break
		codes = np.asarray(codes)                                       #Converting the list into a numpy array 
		[hist, edges] = np.histogram(codes.ravel(), bins=range(0, 256)) #and converting the codes into a histogram
		hist = hist.astype("float")
		return hist                                                   
