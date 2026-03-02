import os, sys
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

imgdir = "./yalefaces"

X = []
Y = []
first = True
for filename in os.listdir(imgdir):
	parts = filename.split('.')
	if parts[1] != 'txt':
		classID = parts[0][7:9]
		print(classID)
		Y.append(classID)
		im = Image.open(imgdir + "/" + filename)
		im = im.resize((40,40))
		im = np.atleast_2d(np.array(im.getdata(),dtype=np.float64))
		X_flattened = np.reshape(im, (1,1600))
		if len(X) == 0:
			X = X_flattened
		else:
			X = np.append(X,X_flattened,axis=0)
   
   
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=2/3, random_state=0)
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)
X_train_standardized = (X_train - X_train_mean) / X_train_std

# Predict training and testing sample using trained model
X_test_standardized = (X_test - X_train_mean) / X_train_std

tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
tree.fit(X_train_standardized, y_train)
y_test_pred = tree.predict(X_test_standardized)

# Compute class priors, accuracy, and generate confusion matrix
classes, counts = np.unique(y_test_pred, return_counts=True)
priors = counts / len(y_test_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Classes Priors:", classes, priors)
print(f"Testing Accuracy: {accuracy_test}")
print("Testing Confusion Matrix:")
print(conf_matrix_test)    