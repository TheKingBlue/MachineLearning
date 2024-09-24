from sklearn import datasets
import numpy as np

# Import dataset load_iris, put data into X and target into y
X, y = datasets.load_iris(return_X_y=True)

# Set every 1 to 0 and every 2 to 1
for i in range(len(y)):
    if y[i] == 1:
        y[i] = 0
    if y[i] == 2:
        y[i] = 1

def sigmoid(theta):
    ...