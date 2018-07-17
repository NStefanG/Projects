import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import imutils
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def sigmoid(z):
	g=0
	g = 1 / (1 + np.exp(-z))
	
	return g

def costFunctionN(theta, X, y):
	m,n = X.shape
	theta = np.c_[theta]
	
	one = y * np.transpose(np.log( sigmoid( np.dot(X,theta) ) ))
	two = (1-y) * np.transpose(np.log( 1 - sigmoid( np.dot(X,theta) ) ))
	J = -(1./m)*(one+two).sum()
	return J
	
def costFunctionR(theta, X, y, lambdaR):
	
	m,n = X.shape
	theta = np.c_[theta]
	# print(X.shape)
	# print(y.shape)
	# print(theta.shape)
	h = sigmoid(np.dot(X,theta))
	# print(h)
	J = 1./m * np.sum((-y)*np.log(h) - (1-y)*np.log(1-h)) + np.sum((lambdaR/m)*np.power(theta[1:],2))
	# print('Costul este: {}'.format(J))
	
	return J
def gradientN(theta, X, y):
	theta = np.c_[theta]
	m,n = X.shape
	h = sigmoid(np.dot(X,theta))
	grad = 1./m * ((h-y)*X).sum(axis=0)
	
	
	return grad
	
def gradientR(theta, X, y, lambdaR):
	# print('e la grad')
	theta = np.c_[theta]
	m,n = X.shape
	h = sigmoid(np.dot(X,theta))
	# print(h.shape)
	# print(y.shape)
	# print(X.shape)
	# print(theta.shape)
	# print((( float(lambdaR) / m )*theta).shape)
	# print(np.dot((sigmoid(np.dot(X,theta) ) - y).T, X).T.shape)
	grad = (1./m) * np.dot((sigmoid(np.dot(X,theta) ) - y).T, X).T + ( float(lambdaR) / m )*theta
	grad_no_regularization = (1./m) * np.dot((sigmoid( np.dot(X,theta) ) - y).T, X).T
	grad[0] = grad_no_regularization[0]
	
	return grad
	
def predict(theta,X):
	m,n = X.shape
	p = np.zeros((m,1))
	
	sigVal = sigmoid(np.dot(X,theta))
	p = sigVal >=0.5
	
	return p
def mapFeature(X1,X2):
	
	degree = 6
	out = np.ones(( X1.shape[0], sum(range(degree + 2)) )) # could also use ((degree+1) * (degree+2)) / 2 instead of sum
	curr_column = 1
	for i in range(1, degree + 1):
		for j in range(i+1):
			out[:,curr_column] = np.power(X1,i-j) * np.power(X2,j)
			curr_column += 1
	return out

def featureNormalize(X):
	mu = np.mean(X, axis=0)
	
	X_norm = X - mu
	
	sigma = np.std(X_norm, axis = 0, ddof = 1)
	X_norm = X_norm / sigma
	
	return X_norm, mu, sigma
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
	plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
	plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
	plt.xlabel("Threshold")
	plt.legend(loc="upper left")
	plt.ylim([0, 1])

def sliding_window(image, stepSize, windowSize):
	#slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			#yield the current window
			yield(x, y, image[y:y + windowSize[1], x:x+windowSize[0]])

def pyramid(image, scale=1.5, minSize=(28, 28)):
	#yield the original image
	yield image
	
	#keep looping over pyramid
	while True:
		#compute the new dimentions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width = w)
		
		#if the resized image does not meet the supplied minimum
		#size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		
		#yield the next image in the pyramid
		yield image
		
def plot_learning_curves(model, X, y):
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
	train_errors, val_errors = [], []
	for m in range(1, len(X_train)):
		model.fit(X_train[:m], y_train[:m])
		y_train_predict = model.predict(X_train[:m])
		y_val_predict = model.predict(X_val)
		train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
		val_errors.append(mean_squared_error(y_val_predict, y_val))
	plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
	plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")		
		
	
	
	
	
	