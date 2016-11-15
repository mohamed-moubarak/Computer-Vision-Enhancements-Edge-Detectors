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


image1 = "..\\images\\image1.png"
image2 = "..\\images\\image2.png"

contrast_stretching(image1)

# print 'RGB shape: ', img.shape 
# print 'rows: ', img.shape[0]
# print 'columns: ', img.shape[1]
# # print 'channels: ', img1.shape[2]
# print 'Pixel intensity: ', img[240, 240]		
