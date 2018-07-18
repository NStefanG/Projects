import cv2
import time
import argparse
import sklearn
from objdetect import sliding_window, pyramid
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from imutils.video import VideoStream

ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-r", "--picamera", type=int, default=-1, help="whether or not the Raspberry pi camera should be used")
args = vars(ap.parse_args())

# image = cv2.imread(args["image"])
(winW, winH) = (28, 28)
model = joblib.load("model_SGD_FINAL.pkl")

print('[INFO] camera sensor is warming up..')
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
#camera = cv2.VideoCapture(0)
time.sleep(2)

while True:
	found = 0
	start = time.time()

	#(grabbed, image) = camera.read()
	#print(camera.isOpened())
	#print(camera.read())
	#cv2.imshow("igm", image)
	#if not grabbed:
		#break
	image = vs.read()
	image = cv2.flip(image, -1)
	imageGRAY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	imageGRAY = cv2.bitwise_not(imageGRAY)
	#imageGRAY = cv2.medianBlur(imageGRAY, 5)
	#cv2.imshow("Poza", imageGRAY)	
	# loop over the image pyramid
	for resized in pyramid(imageGRAY, scale=1.5):
		# loop over the sliding window for each layer of the pyramid
		for (x,y,window) in sliding_window(resized, stepSize=10, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			#cv2.imshow("Gasit", window)
			#print(window.shape)
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			
			data = window.reshape(1,-1)
			#data_score = model.decision_function(data)
			pred = model.predict(data)
			#print("Scorul este:{}".format(data_score))
			#if data_score > 40216:
			if pred :
				#cv2.rectangle(imageGRAY, (x,y), (x + 5, y + 5), (255,255,255), 2)
				#cv2.imshow("Gasit", imageGRAY)
				found = 5
				print(found)
				break
			
	
		if found == 5:
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
	if cv2.waitKey(1) & 0xFF== ord("q"):
		break
	print(time.time() - start)
#camera.release()
cv2.destroyAllWindows()
vs.stop()
