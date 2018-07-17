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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from functii import plot_precision_recall_vs_threshold, plot_learning_curves
from sklearn.datasets import fetch_mldata

def training():
	mnist = fetch_mldata("MNIST original")
	X, y = mnist["data"], mnist["target"]
	X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
	# dfFin = pd.read_csv('dateTank.csv')
	# X = dfFin.iloc[:,2:].values
	# y = np.vstack([np.ones([3702,1]), np.zeros([3702,1])])
	# print(X.shape, y.shape)
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	# shuffle_index = np.random.RandomState(seed = 42).permutation(60000)
	# X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
	for i in range(1,len(X[36000])):
		if i%28 == 0:
			print(X[36000,i], sep='\n')
			print("\n")
		else:
			print(X[36000,i], end=' ')
	cv2.imshow("cifra", X[36000].reshape(28,28))
	cv2.waitKey(0)
	# scaler = StandardScaler()
	# scaler.fit(X_train)
	# X_train = scaler.transform(X_train)
	# X_test = scaler.transform(X_test)
	
	# y_train_5 = (y_train == 5)
	# y_test_5 = (y_test == 5)
	
	# start = time.time()
	# param_grid = [ {'alpha':[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11],
					# 'max_iter':[100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], 'random_state':[42]}]
	# sgd = SGDClassifier()
	# grid_search = GridSearchCV(sgd, param_grid, cv=3, scoring='precision')
	# grid_search.fit(X_train, y_train_5)
	# print(grid_search.best_params_)
	# print(time.time()-start)
	
	# sgd = SGDClassifier(alpha = 1e-07, max_iter = 2000, random_state = 42)
	# sgd.fit(X_train, y_train_5)
	
	
	# print(cross_val_score(sgd, X_train, y_train, cv=3, scoring="accuracy"))
	
	# y_sc = cross_val_predict(sgd, X_train, y_train_5, cv=3)
	# y_scores = cross_val_predict(sgd, X_train, y_train_5, cv=3, method = "decision_function")
	#1 174 720
	# y_scores95 = (y_scores > 40216)
	# print("Scor precizie:{}".format(precision_score(y_train_5, y_sc)))
	# print("Scor recall:{}".format(recall_score(y_train_5, y_sc)))
	# print("Scor precizie 95%:{}".format(precision_score(y_train_5, y_scores95)))
	# print("Scor recall pr 95%:{}".format(recall_score(y_train_5, y_scores95)))
	# precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
	# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
	# y_test_pred = sgd.predict(X_test)
	# print("Scor precizie test set: {}".format(precision_score(y_test_5, y_test_pred)))
	# print("scor recall test set: {}".format(recall_score(y_test_5, y_test_pred)))
	# average_precision = average_precision_score(y_train_5, y_scores)
	# plt.step(recalls, precisions, color='b', alpha=0.2,
         # where='post')
	# plt.fill_between(recalls, precisions, step='post', alpha=0.2,
                 # color='g')
	# plt.xlabel("Recall")
	# plt.ylabel("Precision")
	# plt.ylim([0.0, 1.05])
	# plt.xlim([0.0, 1.0])
	# plt.title('Precision-Recall: AP={0:0.2f}'.format(
          # average_precision))

	# plt.show()
	
	# joblib.dump(sgd, "model_SGD_FINAL.pkl")
	
training()
	