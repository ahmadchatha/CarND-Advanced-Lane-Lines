import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibration import perform_calibration
from utils import abs_sobel_thresh, mag_thresh, dir_threshold, combine_threshold, hls_select, perspective_transform, find_lines, calculate_radius_dist, plot_lane, prev_fit
from moviepy.editor import VideoFileClip
import warnings

mode = 'images'
#mode = 'video'

# perform camera calibration
mtx, dist = perform_calibration()
# source and destination points
src = np.float32([[599,450], [678,450], [240,715],[1060,715]])
dst = np.float32([[355,0],   [940,0],  [355,715],[940,715]])
lfit = None
rfit = None

# pipeline to process an image
def process_image(img, mtx, dist, lfit=None, rfit=None, file_name=None):
	orig_img = img
	# undistort image
	img = cv2.undistort(img, mtx, dist, None, mtx)
	#mpimg.imsave('output_images/'+file_name+ "_undistorted.jpg", img)
	# thresholding
	img = combine_threshold(img)
	#mpimg.imsave('output_images/'+file_name+ "_thresholded.jpg", img, cmap='gray')
	# perspective transform
	img = perspective_transform(img, src, dst)

	#histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
	#plt.plot(histogram)
	#plt.savefig('output_images/'+file_name+ "_histogram.png")
	#plt.close()
	#mpimg.imsave('output_images/'+file_name+ "_transformed.jpg", img, cmap='gray')
	# Detect lane pixels and fit to find the lane boundary
	if lfit is None and rfit is None:
		lfit, rfit = find_lines(img)
	else:
		lfit, rfit = prev_fit(img, lfit, rfit)
	# plotting the lanes on the image
	ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
	left_fitx = lfit[0] * ploty ** 2 + lfit[1] * ploty + lfit[2]
	right_fitx = rfit[0] * ploty ** 2 + rfit[1] * ploty + rfit[2]
	out_img = plot_lane(orig_img, left_fitx, ploty, right_fitx)
	#mpimg.imsave('output_images/'+file_name+ "_lane_plot.jpg", out_img, cmap='gray')
	# Determine the curvature of the lane and vehicle position with respect to center.
	radius, distance = calculate_radius_dist(ploty, left_fitx, right_fitx)
	# Warp the detected lane boundaries back onto the original image.
	out_img = perspective_transform(out_img, dst, src)
	out_img = cv2.addWeighted(orig_img, .5, out_img, .5, 0.0, dtype=0)
	# adding text on the image
	cv2.putText(out_img, "Curve radius: "+ "{:04.2f}".format(radius) + "m", (40, 70), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255))
	cv2.putText(out_img, "Distance from center: "+"{:04.3f}".format(distance) + "m ", (40, 120), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255))
	#mpimg.imsave('output_images/'+file_name+ "_result.jpg", out_img, cmap='gray')
	return out_img, lfit, rfit

def process_video(img):
	global lfit, rfit
	output, lfit, rfit = process_image(img, mtx, dist, lfit, rfit)
	return output



warnings.simplefilter("ignore")
if mode == 'images':
	images = glob.glob('test_images/*.jpg')
	for image in images:
		file_name = os.path.basename(image).split('.')[0]
		print(file_name)
		img = cv2.imread('test_images/'+file_name+'.jpg')
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		mpimg.imsave('output_images/'+file_name+ "_orig.jpg", img)
		process_image(img, mtx, dist, file_name=file_name)
else:
	video = VideoFileClip("project_video.mp4")
	output_video= video.fl_image(process_video)
	output_video.write_videofile("project_video_output.mp4", audio=False)

