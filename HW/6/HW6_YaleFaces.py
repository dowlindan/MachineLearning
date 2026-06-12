import os, sys
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import CategoricalNB, GaussianNB

#Yale Faces Dataset
imgdir = "./yalefaces"

# Read in all images
X = []
Y = []
first = True
for filename in os.listdir(imgdir):
	parts = filename.split('.')
	if parts[1] != 'txt':
		im = Image.open(imgdir + "/" + filename)
		# Resize to (40,40)
		im = im.resize((40,40), Image.NEAREST)
		#im = np.atleast_2d(np.array(im.getdata(),dtype=np.float64))
		im = np.atleast_2d(np.array(im.get_flattened_data(), dtype=np.float64))
		#Reshape into 1x1600 feature vectors
		X_flattened = np.reshape(im, (1,1600))

		# Add to X and Y
		if len(X) == 0:
			X = im
		else:
			X = np.append(X,X_flattened,axis=0)

		classID = parts[0][7:9]
		Y.append(classID)

X_binned = X//32

X_train, X_test, y_train, y_test = train_test_split(X_binned, Y, train_size=2/3, random_state=42, stratify=Y)

# Predict training and testing sample using trained model

classifier = CategoricalNB(min_categories=8)
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)
	

#Compute class priors, accuracy, and generate confusion matrix
classes, counts = np.unique(y_test, return_counts=True)
priors = counts / len(y_test)
print("Actual Classes Priors:\n", classes, "\n", priors)

classes_pred, counts_pred = np.unique(y_test_pred, return_counts=True)
priors_pred = counts_pred / len(y_test_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Prediction Classes Priors:\n", classes_pred, "\n", priors_pred)
print(f"Testing Accuracy: {accuracy_test}")
print("Testing Confusion Matrix:")
print(conf_matrix_test)    