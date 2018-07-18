from __future__ import division
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import cv2
import datetime
import time
import Adafruit_PCA9685

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help = "path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
ap.add_argument("-r", "--picamera", type=int, default=-1, help="whether or not the Raspberry pi camera should be used")
args = vars(ap.parse_args())


greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
#galben
#greenLower = (16, 0, 230)
#greenUpper = (45, 20, 255)
#red
#greenLower = (0, 100, 100)
#greenUpper = (20, 255, 255)
#orange
#greenLower = (10, 0, 230)
#greenUpper = (15, 20, 255)
#blue
#greenLower = (57, 68, 0)
#greenUpper = (151, 255, 255)
#pts = deque(maxlen=args["buffer"])

dt = 1 #timpul de esantionare
u = 0.005 #gradul de crestere al acceleratiei
x,y=0,0

#Obi_zAcc = 0.001 #variatia acceleratiei obiectului/deviatie standard mica
Obi_zAcc = 0.1 

zgX = 1;
zgY = 1;#variatii pt depalsare

Ez = np.matrix([[zgX, 0],
		[0, zgY]])
Ex = np.matrix([[(dt**4)/4, 0, (dt**3)/2, 0],
		[0, (dt**4)/4, 0, (dt**3)/2],
		[(dt**3)/2, 0, dt**2, 0],
		[0, (dt**3)/2, 0, dt**2]])
Ex = Ex*(Obi_zAcc**2)

P=Ex #estimarea pozitiei initiale (matricea de covarianta)

A = np.matrix([[1, 0, dt, 0],
		[0, 1, 0, dt],
		[0, 0, 1, 0],
		[0, 0, 0, 1]]) #matricea de stare a procesului

B = np.matrix([[(dt**2)/2],
		[(dt**2)/2],
		[dt],
		[dt]]) #matricea de stare a comenzii

C = np.matrix([[1, 0, 0, 0],
		[0, 1, 0, 0]]) #masuram numai pozitia

#variabilele pentru rezultat
Q_loc = [] #miscarea reala a obj
vel = [] #viteza reala a obj
Q_loc_mas = np.zeros((2,1)) #traiectoria rezultata din algoritmul de detectie

radius = 0
#variabile de estimare

Q_loc_est = []
vel_estimate = []
P_estimate = P
predic_state = np.zeros((1,1))
predic_var = np.zeros((4,4))

#Q_loc_mas = np.hstack((Q_loc_mas,[[x], [y]]))

Q = np.array([[x], [y], [0], [0]]) #starea initiala
Q_estimat = Q

#scrierea in fisier
#f = open('datePlot2_disptinta', 'w')
printare=0

centruX = 150
centruY = 200
pwmInitOriz = 400
pwmInitVert = 400 
#servo init

pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)

#if not args.get("video", False):
	#camera = cv2.VideoCapture(0)
#else:
	#camera = cv2.VideoCapture(args["video"])
print('[INFO] camera sensor is warming up..')
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

while True:
	#(grabbed, frame) = camera.read()
	frame = vs.read()
	frame = cv2.flip(frame, -1)
	#if args.get("video") and not grabbed:
		#break
	#print(frame)
	frame = imutils.resize(frame, width=400)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None


	if len(cnts) > 0:

		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		#M = cv2.moments(c)
		#center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		mas = np.matrix([[x],
				[y]])
		if radius > 10:
			cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), -1)
			#cv2.circle(frame, center, 5, (0, 0, 255), -1)
	else:
		x,y=np.nan,np.nan
		mas = np.matrix([[x],
				[y]])
	

	 #estimarea starii urmatoare
	Q_estimat = np.dot(A, Q_estimat) + B*u
	predic_state = np.vstack((predic_state, Q_estimat[0,0]))
	 #estimarea urmatoarei covariante
	P = np.dot(np.dot(A, P), A.transpose()) + Ex
	predic_var = np.vstack((predic_var, P))
	
	#Kalman gain 
	#K = P*C'*inv(C*P*C' + Ez)

	a = np.dot(C, P)
	b = np.dot(a, C.transpose())
	c = b + Ez
	d = np.linalg.inv(c)
	e = np.dot(P, C.transpose())
	K = np.dot(e, d)

	

	#updatarea estimarii
	if not np.any(np.isnan(mas)):
		Q_estimat = Q_estimat + np.dot(K, (mas - np.dot(C, Q_estimat)))
		#print("IF_ULLLLLLLLL")

	#updatarea estimarii covariantei

	P = np.dot((np.eye(4) - np.dot(K, C)), P)
	
	cv2.circle(frame, (int(Q_estimat[0,0]), int(Q_estimat[1,0])), int(radius), (0, 0, 255), -1)
	#cv2.circle(frame, center, 5, (0, 0, 255), -1)

	#grad = pixel2degree([int(Q_estimat[0,0]), int(Q_estimat[1,0])])
	
	#pixelX = int((((grad[0] - 0) * (600 - 200)) / (170 - 0)) + 200)
	#pixelY = int((((grad[1] - 0) * (500 - 300)) / (70 - 0)) + 300)
	
	#pixelX = int((((int(Q_estimat[0,0]) - 0) * (600 - 200)) / (300 - 0)) + 200)
	#pixelY = int((((int(Q_estimat[1,0]) - 0) * (500 - 300)) / (400 - 0)) + 300)
	
	if int(Q_estimat[0,0]) > 160:
		pwm.set_pwm(1,0,pwmInitOriz - 5)
		pwmInitOriz -= 5
	elif int(Q_estimat[0,0]) < 140:
		pwm.set_pwm(1,0,pwmInitOriz + 5)
		pwmInitOriz += 5
	if int(Q_estimat[1,0]) > 210:
		pwm.set_pwm(2,0,pwmInitVert + 5)
		pwmInitVert += 5
	elif int(Q_estimat[1,0]) < 190:
		pwm.set_pwm(2,0,pwmInitVert - 5)
		pwmInitVert -= 5
	#if int(Q_estimat[0,0])==150 and int(Q_estimat[1,0])==200:
		#pwm.set_pwm(1,0,0)
		#pwm.set_pwm(2,0,0)
	time.sleep(0.05)
	
	#print("avem PixelX {}, PixelY {}, PwmX {}, PwmY {}". format(int(Q_estimat[0,0]), int(Q_estimat[1,0]), pixelX,pixelY))
	#pts.appendleft(center)

	#for i in range(1, len(pts)):
		#if pts[i - 1] is None or pts[i] is None:
			#continue

		#thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		#cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)gr
	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	#print("x=",x)
	#print("y=",y)
	#print(int(Q_estimat[0,0]))
	#print(int(Q_estimat[1,0]))
	
	#if key == ord("p"):
		#printare = 1
	#if printare == 1:
		#f.write('{}  {}  {}  {}\n'.format(x,y,int(Q_estimat[0,0]),int(Q_estimat[1,0])))

	if key == ord("q"):
		break
	#fps.update()


#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#f.close()
#camera.release()
cv2.destroyAllWindows()
vs.stop()










