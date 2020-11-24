import numpy as np
import math
import cv2
import time
from matplotlib import pyplot as plt
import statistics

class customMe:
	def __init__(self, block_size):
		self.block_size=block_size


	def getCode(self,slice):
		thresh=slice[1,1]
		sliceCopy = slice
		sliceCopy=np.where(sliceCopy>=thresh,1,0)
		code=sliceCopy[0,0]*128 +sliceCopy[0,1]*64 +sliceCopy[0,2]*32 +sliceCopy[1,2]*16 +sliceCopy[2,2]*8 +sliceCopy[2,1]*4 +sliceCopy[2,0]*2+sliceCopy[1,0]
		return code

	def findLBP(self,img):
		# t = 5
		# blocks=self.divide_img(img)
		# num_blocks=len(blocks)
		# histograms=np.zeros((num_blocks,255))
		# for i in range(0,num_blocks):
		# 	hist=self.hist_for_block(blocks[i])
		# 	histograms[i,:]=hist
		# final=histograms.flatten()
		# final/=(histograms.sum())
		(N, M) = img.shape
		codes=[]
		r = 0
		while True:
			c = 0
			while True:
				sliceMe = img[r:r + 3, c:c + 3]
				thresh = sliceMe[1, 1]
				sliceCopy = sliceMe
				sliceCopy = np.where(sliceCopy >= thresh, 1, 0)
				code = sliceCopy[0, 0] * 128 + sliceCopy[0, 1] * 64 + sliceCopy[0, 2] * 32 + sliceCopy[1, 2] * 16 + \
				       sliceCopy[2, 2] * 8 + sliceCopy[2, 1] * 4 + sliceCopy[2, 0] * 2 + sliceCopy[1, 0]
				codes.append(code)
				c = c + 3
				if c + 2 > M - 1:
					break
			r = r + 3
			if r + 2 > N - 1:
				break
		codes = np.asarray(codes)
		[hist, edges] = np.histogram(codes.ravel(), bins=range(0, 256))
		hist = hist.astype("float")
		return hist

	# def divide_img(self, img):
	# 	(N,M)=img.shape
	# 	blocks = []
	# 	x = 0
	# 	for i in range(0, N+1):
	# 		y = 0
	# 		for j in range(0, M+1):
	# 			block=img[x:x+self.block_size, y:y+self.block_size]
	# 			blocks.append(block)
	# 			y += self.block_size
	# 			if y+self.block_size > M-1:
	# 				break
	# 		x+=self.block_size
	# 		if x +self.block_size> N-1:
	# 			break
	# 	return blocks

	def hist_for_block(self,img):
		codes = []
		(N, M) = img.shape
		r = 0
		while True:
			c = 0
			while True:
				sliceMe = img[r:r + 3, c:c + 3]
				thresh = sliceMe[1, 1]
				sliceCopy = sliceMe
				sliceCopy = np.where(sliceCopy >= thresh, 1, 0)
				code = sliceCopy[0, 0] * 128 + sliceCopy[0, 1] * 64 + sliceCopy[0, 2] * 32 + sliceCopy[1, 2] * 16 + sliceCopy[2, 2] * 8 + sliceCopy[2, 1] * 4 + sliceCopy[2, 0] * 2 + sliceCopy[1, 0]
				codes.append(code)
				c = c + 3
				if c + 2 > M - 1:
					break
			r = r + 3
			if r + 2 > N - 1:
				break
		codes = np.asarray(codes)
		[hist, edges] = np.histogram(codes.ravel(), bins=range(0, 256))
		hist = hist.astype("float")
		return hist
#
#
# testImg = "F:\\TestProject\\Misc\\sideMe.jpg"
#
# img = cv2.imread(testImg,cv2.IMREAD_GRAYSCALE)
# des = customMe(16)
# tic=time.time()
# hist=des.findLBP(img[0:300,0:300])
# toc=time.time()
# print(toc-tic)
# print(hist.shape)
