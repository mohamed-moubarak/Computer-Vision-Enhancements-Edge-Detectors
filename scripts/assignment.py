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
	Gximg = img.copy()
	Gyimg = img.copy()
	new = img.copy()
	for row in xrange(1,padded_image.shape[0]-1):
		for column in xrange(1,padded_image.shape[1]-1):
			Gx = (padded_image[row - 1, column + 1] + (2.0 * padded_image[row, column + 1]) + padded_image[row + 1, column + 1]) - (padded_image[row - 1, column - 1] + (2.0 * padded_image[row, column - 1]) + padded_image[row + 1, column - 1])
			Gximg[row - 1, column - 1] = math.fabs(Gx)
			Gy = (padded_image[row - 1, column - 1] + (2.0 * padded_image[row - 1, column]) + padded_image[row - 1, column + 1]) - (padded_image[row + 1, column - 1] + (2.0 * padded_image[row + 1, column]) + padded_image[row + 1, column + 1])
			Gyimg[row - 1, column - 1] = math.fabs(Gy)
			new[row - 1, column - 1] = math.fabs(Gx) + math.fabs(Gy)

	cv2.imwrite('..\\results\\sobel_edge_detection.png',new)
	cv2.imwrite('..\\results\\sobel_edge_detection_Gx.png',Gximg)
	cv2.imwrite('..\\results\\sobel_edge_detection_Gy.png',Gyimg)
	
def prewitt_edge_detection(filepath):
	img = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	padded_image = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT,value = 0)
	new = img.copy()

	for row in xrange(1,padded_image.shape[0]-1):
		for column in xrange(1,padded_image.shape[1]-1):
			kernelList = []

			kernel1 = (-1 * padded_image[row - 1, column - 1]) + (1 * padded_image[row - 1, column]) + (1 * padded_image[row - 1, column + 1]) + (-1 * padded_image[row, column - 1]) + (-2 * padded_image[row, column]) + (1 * padded_image[row, column + 1]) + (-1 * padded_image[row + 1, column - 1]) + (1 * padded_image[row + 1, column]) + (1 * padded_image[row + 1, column + 1])

			kernel2 = (1 * padded_image[row - 1, column - 1]) + (1 * padded_image[row - 1, column]) + (1 * padded_image[row - 1, column + 1]) + (-1 * padded_image[row, column - 1]) + (-2 * padded_image[row, column]) + (1 * padded_image[row, column + 1]) + (-1 * padded_image[row + 1, column - 1]) + (-1 * padded_image[row + 1, column]) + (1 * padded_image[row + 1, column + 1])

			kernel3 = (1 * padded_image[row - 1, column - 1]) + (1 * padded_image[row - 1, column]) + (1 * padded_image[row - 1, column + 1]) + (1 * padded_image[row, column - 1]) + (-2 * padded_image[row, column]) + (1 * padded_image[row, column + 1]) + (-1 * padded_image[row + 1, column - 1]) + (-1 * padded_image[row + 1, column]) + (-1 * padded_image[row + 1, column + 1])

			kernel4 = (1 * padded_image[row - 1, column - 1]) + (1 * padded_image[row - 1, column]) + (1 * padded_image[row - 1, column + 1]) + (1 * padded_image[row, column - 1]) + (-2 * padded_image[row, column]) + (-1 * padded_image[row, column + 1]) + (1 * padded_image[row + 1, column - 1]) + (-1 * padded_image[row + 1, column]) + (-1 * padded_image[row + 1, column + 1])

			kernel5 = (1 * padded_image[row - 1, column - 1]) + (1 * padded_image[row - 1, column]) + (-1 * padded_image[row - 1, column + 1]) + (1 * padded_image[row, column - 1]) + (-2 * padded_image[row, column]) + (-1 * padded_image[row, column + 1]) + (1 * padded_image[row + 1, column - 1]) + (1 * padded_image[row + 1, column]) + (-1 * padded_image[row + 1, column + 1])

			kernel6 = (1 * padded_image[row - 1, column - 1]) + (-1 * padded_image[row - 1, column]) + (-1 * padded_image[row - 1, column + 1]) + (1 * padded_image[row, column - 1]) + (-2 * padded_image[row, column]) + (-1 * padded_image[row, column + 1]) + (1 * padded_image[row + 1, column - 1]) + (1 * padded_image[row + 1, column]) + (1 * padded_image[row + 1, column + 1])

			kernel7 = (-1 * padded_image[row - 1, column - 1]) + (-1 * padded_image[row - 1, column]) + (-1 * padded_image[row - 1, column + 1]) + (1 * padded_image[row, column - 1]) + (-2 * padded_image[row, column]) + (1 * padded_image[row, column + 1]) + (1 * padded_image[row + 1, column - 1]) + (1 * padded_image[row + 1, column]) + (1 * padded_image[row + 1, column + 1])

			kernel8 = (-1 * padded_image[row - 1, column - 1]) + (-1 * padded_image[row - 1, column]) + (1 * padded_image[row - 1, column + 1]) + (-1 * padded_image[row, column - 1]) + (-2 * padded_image[row, column]) + (1 * padded_image[row, column + 1]) + (1 * padded_image[row + 1, column - 1]) + (1 * padded_image[row + 1, column]) + (1 * padded_image[row + 1, column + 1])

			kernelList.extend([math.fabs(kernel1), math.fabs(kernel2), math.fabs(kernel3), math.fabs(kernel4), math.fabs(kernel5), math.fabs(kernel6), math.fabs(kernel7), math.fabs(kernel8)])
			new[row - 1, column - 1] = max(kernelList)


	cv2.imwrite('..\\results\\prewitt_edge_detection.png',new)

def canny_edge_detection(filepath, t1, t2):
	img = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	padded_image = cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_CONSTANT,value = 0)
	gaussian = img.copy()
	canny = img.copy()

	for row in xrange(2,padded_image.shape[0]-2):
		for column in xrange(2,padded_image.shape[1]-2):
			kernel = (1 * padded_image[row - 2, column - 2]) + (4 * padded_image[row - 2, column - 1]) + (7 * padded_image[row - 2, column]) + (4 * padded_image[row - 2, column + 1]) + (1 * padded_image[row - 2, column + 2]) + (4 * padded_image[row - 1, column - 2]) + (16 * padded_image[row - 1, column - 1]) + (26 * padded_image[row - 1, column]) + (16 * padded_image[row - 1, column + 1]) + (4 * padded_image[row - 1, column + 2]) + (7 * padded_image[row, column - 2]) + (26 * padded_image[row, column - 1]) + (40 * padded_image[row, column]) + (26 * padded_image[row, column + 1]) + (7 * padded_image[row, column + 2]) + (4 * padded_image[row + 1, column - 2]) + (16 * padded_image[row + 1, column - 1]) + (26 * padded_image[row + 1, column]) + (16 * padded_image[row + 1, column + 1]) + (4 * padded_image[row + 1, column + 2]) + (1 * padded_image[row + 2, column - 2]) + (4 * padded_image[row + 2, column - 1]) + (7 * padded_image[row + 2, column]) + (4 * padded_image[row + 2, column + 1]) + (1 * padded_image[row + 2, column + 2])

			gaussian[row - 2, column - 2] = kernel / 272

	# cv2.imwrite('..\\results\\gaussian_filter.png',gaussian)

	# prewitt_edge_detection('..\\results\\gaussian_filter.png')
	# canny = cv2.imread('..\\results\\gaussian_filter.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)

	for row in xrange(0,canny.shape[0]):
		for column in xrange(0,canny.shape[1]):
			if row == 0 or row == canny.shape[0] - 1 or column == 0 or column == canny.shape[1] - 1 or canny[row, column] < t2:
				canny[row, column] = 0
			else:
				if not (canny[row - 1, column - 1] > 0 or canny[row - 1, column] > 0 or canny[row - 1, column + 1] > 0 or canny[row, column - 1] > 0 or canny[row, column + 1] > 0 or canny[row + 1, column - 1] > 0 or canny[row + 1, column] > 0 or canny[row + 1, column + 1] > 0):
					canny[row, column] = 0
				

	
	cv2.imwrite('..\\results\\canny_edge_detection.png',canny)
	



image1 = "..\\images\\image1.png"
image2 = "..\\images\\image2.png"

contrast_stretching(image1)
histogram_equalization(image1)
sobel_edge_detection(image2)
prewitt_edge_detection(image2)
canny_edge_detection(image2, 150, 75)

# print 'RGB shape: ', img.shape 
# print 'rows: ', img.shape[0]
# print 'columns: ', img.shape[1]
# # print 'channels: ', img1.shape[2]
# print 'Pixel intensity: ', img[240, 240]		
