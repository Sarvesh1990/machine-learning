#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

with open("ex1data1python.txt") as f:
    data = f.read()

data = data.split('\n')

x = [row.split(' ')[0] for row in data]
y = [row.split(' ')[1] for row in data]	

#plotData(x, y);

X = np.matrix(x).transpose();
Y = np.matrix(y).transpose();
X = X.astype(float);
Y = Y.astype(float);
m = len(y)
print(m);
X = np.c_[np.ones(m), X];

theta = np.zeros((2, 1));
thetaTemp = np.zeros((2, 1));

#Some gradient descent settings
iterations = 1500;
alpha = 0.01;

def computeCost(X, Y, theta) :
	J = 0;
	Cost = 0;

	for i in range (0,m) : 
		XI = np.matrix(X[i , :]).transpose();
		YI = np.matrix(Y[i, :]);
		dot = np.dot(theta.transpose(), XI);
		temp = np.square((dot - YI));
		Cost = Cost + temp;

	J = (Cost/(2*m));
	return J;

J = computeCost(X, Y, theta);

for iter in range (0, iterations) : 
	for j in range(0, theta.shape[0]) :
		Cost = 0;
		for i in range (0, m) :
			XI =  np.matrix(X[i , :]).transpose();
			YI = np.matrix(Y[i, :]);
			dot = np.dot(theta.transpose(), XI);
			temp = (dot - YI)*XI[j];
			Cost = Cost + temp;
		thetaTemp[j] = theta[j] - (alpha * Cost)/m;

	theta = thetaTemp;


print (theta);

def plotData(x, y) :
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.set_title("Plot title...")    
	ax1.set_xlabel('your x label..')
	ax1.set_ylabel('your y label...')

	ax1.plot(x,y, 'ro', label='the data')

	leg = ax1.legend()

	plt.show()