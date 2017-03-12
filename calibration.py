import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

images = glob.glob('camera_cal/calibration*.jpg')

def perform_calibration():
	"""
	perform camera calibration. compute dist and mtx.
	"""
	# Prepare object points like (0,0,0), (1,0,0) ...
	objp = np.zeros((6 * 9, 3), np.float32)
	objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # x,y coordinates

	for image in images:
		img = cv2.imread(image)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

		if ret == True:
		    imgpoints.append(corners)
		    objpoints.append(objp)

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 							gray.shape[::-1], None, None)

	return mtx, dist

def test():
	mtx, dist = perform_calibration()
	img1 = cv2.imread('camera_cal/calibration1.jpg')
	out1 = cv2.undistort(img1, mtx, dist, None, mtx)
	cv2.imwrite('output_images/calibrated1.jpg', out1)

test()