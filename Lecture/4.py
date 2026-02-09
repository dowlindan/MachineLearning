import numpy as np
x = np.array([[2,5]])
y = np.array([[3]])

# Add bias feature
x = np.concatenate((x, np.ones((x.shape[0],1))),axis=1)
print(x)

# Initialize weights to small random numbers

# Online - Only updating 1 obs at a time
w = np.array([[1],[1],[1]])
print(w)

yhat1 = x@w
print(yhat1)

J1 = (y-yhat1)**2
print(J1)

dJdw = 2*(yhat1-y)*x.T

w = w - dJdw
