from __future__ import division
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



image1 = "..\\images\\image1.png"
image2 = "..\\images\\image2.png"

contrast_stretching(image1)
histogram_equalization(image1)

# print 'RGB shape: ', img.shape 
# print 'rows: ', img.shape[0]
# print 'columns: ', img.shape[1]
# # print 'channels: ', img1.shape[2]
# print 'Pixel intensity: ', img[240, 240]		
