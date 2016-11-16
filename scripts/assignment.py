from __future__ import division
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


def contrast_stretching(filepath):
	img = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	minIntensity = img[0,0]
	maxIntensity = img[0,0]

	for row in xrange(0,img.shape[0]):
		for column in xrange(0,img.shape[1]):
			if img[row, column] < minIntensity:
				minIntensity = img[row, column]
			if img[row, column] > maxIntensity:
				maxIntensity = img[row, column]

	scalingFactor = 255/(maxIntensity - minIntensity)

	for row in xrange(0,img.shape[0]):
		for column in xrange(0,img.shape[1]):
			img[row, column] = (img[row, column] - minIntensity) * scalingFactor

	cv2.imwrite('..\\results\\contrast_stretching.png',img)

def histogram_equalization(filepath):
	intensitiesList = []
	frequencyList = []
	cumulativeFrequencyList = []
	
	img = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	numberOfPixels = img.shape[0] * img.shape[1]

	for row in xrange(0,img.shape[0]):
		for column in xrange(0,img.shape[1]):
			if img[row, column] not in intensitiesList:
				intensitiesList.extend([img[row, column]])

	print 'Intensities in the image: ', str(intensitiesList)
	
	intensitiesList.sort()
	print 'Sorted intensities in the image: ', str(intensitiesList)

	for intensity in intensitiesList:
		count = 0
		for row in xrange(0,img.shape[0]):
			for column in xrange(0,img.shape[1]):
				if intensity == img[row, column]:
					count += 1

		frequencyList.extend([count])
		if not cumulativeFrequencyList:
			cumulativeFrequencyList.extend([count])
		else:
			cumulativeValue = cumulativeFrequencyList[len(cumulativeFrequencyList)-1] + count
			cumulativeFrequencyList.extend([cumulativeValue])

	print 'Frequencies of each intensity (Sorted intensities): ', str(frequencyList)

	print 'Cumulative frequencies: ', str(cumulativeFrequencyList)

	for row in xrange(0,img.shape[0]):
		for column in xrange(0,img.shape[1]):
			i = intensitiesList.index(img[row, column])
			img[row, column] = int(((cumulativeFrequencyList[i] - cumulativeFrequencyList[0])/(numberOfPixels - cumulativeFrequencyList[0])) * (255))

	cv2.imwrite('..\\results\\histogram_equalization.png',img)

def sobel_edge_detection(filepath):
	img = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	padded_image = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT,value = 0)
	Gximg = img
	Gyimg = img
	new = img

	for row in xrange(1,padded_image.shape[0]-1):
		for column in xrange(1,padded_image.shape[1]-1):
			Gx = (padded_image[row - 1, column + 1] + (2.0 * padded_image[row, column + 1]) + padded_image[row + 1, column + 1]) - (padded_image[row - 1, column - 1] + (2.0 * padded_image[row, column - 1]) + padded_image[row + 1, column - 1])
			Gximg[row - 1, column - 1] = math.fabs(Gx)
			Gy = (padded_image[row - 1, column - 1] + (2.0 * padded_image[row - 1, column]) + padded_image[row - 1, column + 1]) - (padded_image[row + 1, column - 1] + (2.0 * padded_image[row + 1, column]) + padded_image[row + 1, column + 1])
			Gyimg[row - 1, column - 1] = math.fabs(Gy)
			new[row - 1, column - 1] = math.fabs(Gx) + math.fabs(Gy)
	# row = 200
	# column = 100
	# Gx = (padded_image[row - 1, column + 1] + 2 * padded_image[row, column + 1] + padded_image[row + 1, column + 1]) - (padded_image[row - 1, column - 1] + 2 * padded_image[row, column - 1] + padded_image[row + 1, column - 1])
	# Gximg[row - 1, column - 1] = Gx		
	# Gy = (padded_image[row - 1, column - 1] + 2 * padded_image[row - 1, column] + padded_image[row - 1, column + 1]) - (padded_image[row + 1, column - 1] + 2 * padded_image[row + 1, column] + padded_image[row + 1, column + 1])
	# Gyimg[row - 1, column - 1] = Gy
	# img[row - 1, column - 1] = math.sqrt(Gx ** 2 + Gy ** 2)
	# print Gx
	# print Gy
	# print img[row - 1, column - 1]
	cv2.imwrite('..\\results\\sobel_edge_detection.png',new)
	cv2.imwrite('..\\results\\sobel_edge_detection_Gx.png',Gximg)
	cv2.imwrite('..\\results\\sobel_edge_detection_GY.png',Gyimg)
	cv2.imwrite('..\\results\\padded_image.png',padded_image)

	# img = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	# padded_image = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_padded_image)

	# for row in xrange(1,padded_image.shape[0]-1):
	# 	for column in xrange(1,padded_image.shape[1]-1):
			

	# cv2.imwrite('..\\results\\sobel_edge_detection_Gy.png',img)



image1 = "..\\images\\image1.png"
image2 = "..\\images\\image2.png"
image3 = "..\\images\\image3.png"

contrast_stretching(image1)
histogram_equalization(image1)
sobel_edge_detection(image2)

# print 'RGB shape: ', img.shape 
# print 'rows: ', img.shape[0]
# print 'columns: ', img.shape[1]
# # print 'channels: ', img1.shape[2]
# print 'Pixel intensity: ', img[240, 240]		
