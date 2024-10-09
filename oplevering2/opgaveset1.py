from sklearn import datasets
import numpy as np

# Import dataset load_iris, put data into X and target into y
X, target = datasets.load_iris(return_X_y=True)

# Set every 1 to 0 and every 2 to 1
for i in range(len(target)):
    if target[i] == 1:
        target[i] = 0
    if target[i] == 2:
        target[i] = 1
        

# Transform target into a properly shaped y
y = np.expand_dims(target, axis=1)

def sigmoid(h: float) -> float:
    """Applies the sigmoid function to h (The prediction)"""
    # exp calculates the constant e to the power of the given argument
    e = np.exp(-h)
    return np.divide(1, (1+e))

m,n = X.shape

# Create theta and modify X correspondingly
theta = np.ones((1, (n+1))) # Plus 1 to account for theta zero
X = np.c_[np.ones(m), X]

def cost(y, h, m) -> float:
    """Calculates the cost based on y, h and m"""
    return np.sum(y * np.log(h) + (1-y) * np.log(1-h))/m

# Gradiant descent
iterations = 1500
alpha = 0.01
costs = []
for _ in range(iterations):
    h = np.matmul(X, theta.T)
    s = sigmoid(h)
    diff = np.subtract(s, y)
    mul = np.dot(diff.T, X)
    theta -= np.multiply(alpha, np.multiply((1/m), mul))
    costs.append(cost(y, s, m))

print(f"Eerste berekening van de kosten was {costs[0]}, de laatste berekening van de kosten was {costs[-1]}")

import matplotlib.pyplot as plt
plt.plot(costs)
plt.show()