import numpy as np
import cv2
import math

def pixel2degree(centru):
	
	radsX = math.acos((centru[0]*150 + 200*200 + 1*1)/(math.sqrt(centru[0]*centru[0] + 200*200 + 1*1) * math.sqrt(150*150 + 200*200 + 1*1)))
	radsY = math.acos((150*150 + centru[1]*200 + 1*1)/(math.sqrt(150*150 + centru[1]*centru[1] + 1*1) * math.sqrt(150*150 + 200*200 +1*1)))       
	
	grad = [0] * 2
	grad[0] = math.degrees(radsX)
	grad[1] = math.degrees(radsY)

	return grad