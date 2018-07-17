import pandas as pd
import numpy as np
import time
import cv2
import argparse
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from functii import plot_precision_recall_vs_threshold, plot_learning_curves
from sklearn.datasets import fetch_mldata

def training():
	dfFin = pd.read_csv('dateTank.csv')
	# print(dfFin.head())
	X = dfFin.iloc[:,2:].values
	m,n = X.shape
	y = np.vstack([np.ones([1365,1]), np.zeros([1365,1])])
	m1,n1 = y.shape
	y = y.reshape(m1,)
	
	shuffle_index = np.random.RandomState(seed = 42).permutation(m)
	X, y = X[shuffle_index], y[shuffle_index]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	# print(sum(y_train == 1))
	# print(sum(y_train == 0))
	# scaler = StandardScaler()
	# scaler.fit(X_train)
	# X_train = scaler.transform(X_train)
	# X_test = scaler.transform(X_test)
	
	#0.01-100-147881
	# start = time.time()
	# param_grid = [ {'alpha':[100,10,1,1e-1,1e-2,1e-3],
					# 'max_iter':[800, 1000, 1500], 'random_state':[42]}]
	# sgd = SGDClassifier()
	# grid_search = GridSearchCV(sgd, param_grid, n_jobs= 2, cv=3, scoring='precision')
	# grid_search.fit(X_train, y_train)
	# print(grid_search.best_params_)
	# print(time.time()-start)
	sgd = SGDClassifier(alpha = 10,max_iter = 1000, random_state = 42)
	sgd.fit(X_train, y_train)
	
	
	# print(cross_val_score(sgd, X_train, y_train, cv=3, scoring="accuracy"))
	
	y_sc = cross_val_predict(sgd, X_train, y_train, cv=3)
	y_scores = cross_val_predict(sgd, X_train, y_train, cv=3, method = "decision_function")
	#1 174 720
	# y_scores95 = (y_scores > 128575)
	print("Scor precizie:{}".format(precision_score(y_train, y_sc)))
	print("Scor recall:{}".format(recall_score(y_train, y_sc)))
	# print("Scor precizie:{}".format(precision_score(y_train, y_scores95)))
	# print("Scor recall:{}".format(recall_score(y_train, y_scores95)))
	precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
	plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
	y_test_pred = sgd.predict(X_test)
	print("Scor precizie test set: {}".format(precision_score(y_test, y_test_pred)))
	print("scor recall test set: {}".format(recall_score(y_test, y_test_pred)))
	plt.show()
	joblib.dump(sgd, "modelTank_SGD.pkl")
	
	
if __name__ == '__main__':	
	training()
	