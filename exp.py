import cv2
import time
import argparse
from functii import sliding_window, pyramid
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

# image = cv2.imread(args["image"])
(winW, winH) = (28, 28)
model = joblib.load("model_SGD.pkl")
camera = cv2.VideoCapture(0)
time.sleep(2)
while True:
	found = 0
	start = time.time()
	(grabbed, image) = camera.read()
	# print(camera.isOpened())
	# print(camera.read())
	cv2.imshow("igm", image)
	if not grabbed:
		break
	imageGRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
	# loop over the image pyramid
	for resized in pyramid(imageGRAY, scale=1.5):
		# loop over the sliding window for each layer of the pyramid
		for (x,y,window) in sliding_window(resized, stepSize=5, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			data = window.flatten()
			data_score = model.decision_function([data])
			if data_score > 137:
				# cv2.rectangle(imageGRAY, (x,y), (x + winW, y + winH), (0,255,0), 2)
				# cv2.imshow("Gasit", imageGRAY)
			
				found = 5
				print(found)
				break
		if found :
			break
			# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
			# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
			# WINDOW
			
			# since we do not have a classifier, we'll just draw the window
			# clone = resized.copy()
			# cv2.rectangle(clone, (x,y), (x + winW, y + winH), (0,255,0), 2)
			# cv2.imshow("WIndow", clone)
			# cv2.waitKey(1)
			# time.sleep(0.025)
	key = cv2.waitKey(1) & 0xFF	
	if key == ord("q"):
		break		
	print(time.time() - start)