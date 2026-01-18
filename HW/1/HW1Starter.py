import os, sys
from PIL import Image
import numpy
import matplotlib.pyplot as plt

imgdir = "../../../datasets/yalefaces"

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

print(X.shape)
plt.imshow(numpy.reshape(X[0],(40,40)),cmap='gray')
plt.show()
