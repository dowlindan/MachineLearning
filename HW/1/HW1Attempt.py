import os, sys
from PIL import Image
import numpy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

imgdir = "./yalefaces"

X = []
first = True
for filename in os.listdir(imgdir):
	parts = filename.split('.')
	if parts[1] != 'txt':
		im = Image.open(imgdir + "/" + filename)
		im = im.resize((40,40))
		im = numpy.atleast_2d(numpy.array(im.getdata(),dtype=numpy.float64))
		if len(X) == 0:
			X = im
		else:
			X = numpy.append(X,im,axis=0)

#print(X.shape)
#plt.imshow(numpy.reshape(X[0],(40,40)),cmap='gray')
#plt.show()

# Perform PCA finding smallest k such that 95% of eigenvalues are retained
pca = PCA(n_components=0.95)
pca.fit(X)

# Project subject02.centerlight onto k most important principal components
subject02centerlightimg = Image.open(imgdir + "/subject02.centerlight")
subject02centerlightimg = subject02centerlightimg.resize((40,40))
subject02centerlight = numpy.atleast_2d(numpy.array(subject02centerlightimg.getdata(),dtype=numpy.float64))

Z = pca.transform(subject02centerlight)

# Reconstruct Z using k most significant principal components, back to 1x1600
Xr = pca.inverse_transform(Z)
Xrimg = numpy.reshape(Xr, (40,40))

# Display original and reconstructed images and output values of k
print("K values: ", pca.n_components_)
print(pca.explained_variance_)

fig = plt.figure()

plt.subplot(1,2,1)
plt.imshow(numpy.reshape(subject02centerlight, (40,40)), cmap='gray')
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(Xrimg, cmap='gray')
plt.title("Reconstructed Image")

plt.show()

