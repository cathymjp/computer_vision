import numpy as np
import cv2
import math
import argparse


parser = argparse.ArgumentParser(description='create test images from raw dicom')
parser.add_argument('--input', help='input image where you want to compute sharpness map', required=True)

args = vars(parser.parse_args())


def im2double(im):
	min_val = np.min(im.ravel())
	max_val = np.max(im.ravel())
	out = (im.astype('float') - min_val) / (max_val - min_val)
	return out


def s(x):
	temp = x>0
	return temp.astype(float)


def lbpCode(im_gray, threshold):
	width, height = im_gray.shape

	interpOff = math.sqrt(2)/2
	I = im2double(im_gray)

	pt = cv2.copyMakeBorder(I, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

	right = pt[1:-1, 2:]
	left = pt[1:-1, :-2]
	above = pt[:-2, 1:-1]
	below = pt[2:, 1:-1];
	aboveRight = pt[:-2, 2:]
	aboveLeft = pt[:-2, :-2]
	belowRight = pt[2:, 2:]
	belowLeft = pt[2:, :-2]

	interp0 = right
	interp1 = (1-interpOff)*((1-interpOff) * I + interpOff * right) + interpOff *((1-interpOff) * above + interpOff * aboveRight)

	interp2 = above
	interp3 = (1-interpOff)*((1-interpOff) * I + interpOff * left) + interpOff *((1-interpOff) * above + interpOff * aboveLeft)

	interp4 = left
	interp5 = (1-interpOff)*((1-interpOff) * I + interpOff * left) + interpOff *((1-interpOff) * below + interpOff * belowLeft)

	interp6 = below
	interp7 = (1-interpOff)*((1-interpOff) * I + interpOff * right) + interpOff *((1-interpOff) * below + interpOff * belowRight)

	s0 = s(interp0 - I-threshold)
	s1 = s(interp1 - I-threshold)
	s2 = s(interp2 - I-threshold)
	s3 = s(interp3 - I-threshold)
	s4 = s(interp4 - I-threshold)
	s5 = s(interp5 - I-threshold)
	s6 = s(interp6 - I-threshold)
	s7 = s(interp7 - I-threshold)

	LBP81 = s0 * 1 + s1 * 2+s2 * 4   + s3 * 8+ s4 * 16  + s5 * 32  + s6 * 64  + s7 * 128
	LBP81.astype(int)

	U = np.abs(s0 - s7) + np.abs(s1 - s0) + np.abs(s2 - s1) + np.abs(s3 - s2) + np.abs(s4 - s3) + np.abs(s5 - s4) \
		+ np.abs(s6 - s5) + np.abs(s7 - s6)
	LBP81riu2 = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
	LBP81riu2[U > 2] = 9

	return LBP81riu2


def lbpSharpness(im_gray, s, threshold):
	lbpmap  = lbpCode(im_gray, threshold)
	window_r = (s-1)//2
	h, w = im_gray.shape[:2]
	map =  np.zeros((h, w), dtype=float)
	lbpmap_pad = cv2.copyMakeBorder(lbpmap, window_r, window_r, window_r, window_r, cv2.BORDER_REPLICATE)

	lbpmap_sum = (lbpmap_pad==6).astype(float) + (lbpmap_pad==7).astype(float) + (lbpmap_pad==8).astype(float) \
				 + (lbpmap_pad==9).astype(float)
	integral = cv2.integral(lbpmap_sum)
	integral = integral.astype(float)

	map = (integral[s-1:-1, s-1:-1] - integral[0:h, s-1:-1] - integral[s-1:-1, 0:w] + integral[0:h, 0:w])/math.pow(s,2)

	return map


if __name__=='__main__':
	img = cv2.imread(args['input'], cv2.IMREAD_COLOR)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	result = img.copy()

	################################
	ddepth = cv2.CV_16S
	kernel_size = 3

	# Remove noise by blurring with a Gaussian filter
	img = cv2.GaussianBlur(img, (3, 3), 0)

	# Apply Laplace function
	dst = cv2.Laplacian(img_gray, ddepth, ksize=kernel_size)
	abs_dst = cv2.convertScaleAbs(dst)
	# cv2.imshow("abs", abs_dst)
	ret, thresh2 = cv2.threshold(abs_dst, 127, 255, cv2.THRESH_BINARY_INV)
	# cv2.imshow("dst", thresh2)

	# Line detection
	edges = cv2.Canny(thresh2, 50, 150, apertureSize=3)
	lines = cv2.HoughLines(thresh2, 1, np.pi / 180, 200)
	for rho, theta in lines[0]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a * rho
		y0 = b * rho
		x1 = int(x0 + 1000 * (-b))
		y1 = int(y0 + 1000 * (a))
		x2 = int(x0 - 1000 * (-b))
		y2 = int(y0 - 1000 * (a))

		#cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

	# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
	# detect_horizontal = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
	# cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#
	# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	#
	# edges = cv2.Canny(thresh2, 50, 150, apertureSize=3)
	# minLineLength = 10
	# maxLineGap = 1
	# minNumLines = 15
	# lines = cv2.HoughLinesP(thresh2, 1, np.pi / 180, 100, minLineLength, maxLineGap)
	# for x1, y1, x2, y2 in lines[0]:
	# 	cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
	#
	# for c in cnts:
	# 	cv2.drawContours(result, [c], -1, (36, 255, 8), 1)

	#cv2.imshow('result', result)
	############################

	sharpness_map = lbpSharpness(dst, 21, 0.016)
	sharpness_map = (sharpness_map - np.min(sharpness_map))/(np.max(sharpness_map - np.min(sharpness_map)))

	sharpness_map = (sharpness_map*255).astype('uint8')
	concat = np.concatenate((img, np.stack((sharpness_map,)*3, -1)), axis=1)
	cv2.imshow('img_concat', sharpness_map)

	cv2.waitKey(0)
	cv2.destroyAllWindows()