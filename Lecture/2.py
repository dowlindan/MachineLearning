import numpy as np
X = np.array([0.0],[1,0],[2,0],[3,0])
X[:,1] = X[:,1] + 3

m = np.mean(X, axis=0)
Xz = X - m 

Sigma = (Xz.T @ Xz)/(Xz.shape[0]-1)
print(Sigma)
print(np.cov(Xz, rowvar=False))

[vals,vecs] = np.linalg.eig(Sigma)
print(vecs)
print(vals)
    
#first eigenvalue relates to first column, etc
W = np.atleast_2d(vecs[:,-1]).T
Z = np.atleast_2d(Xz@W)
print(Z)
Xhat = Z@(W.T)
print(Xhat)