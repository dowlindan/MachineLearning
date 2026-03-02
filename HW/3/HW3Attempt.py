
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


dataFilepath = './spambase.data'

# Read in data
data = np.loadtxt(dataFilepath, delimiter=',')

X = data[:, :57]
Y = data[:, 57]

print(X)
print(Y)

# Shuffle and split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=2/3, random_state=0)
 
# Z Score data using training data
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)
X_train_standardized = (X_train - X_train_mean) / X_train_std

# Train logistic classifier on trained model
lr = LogisticRegression()
lr.fit(X_train_standardized, y_train)

# Predict training and testing sample using trained model
X_test_standardized = (X_test - X_train_mean) / X_train_std

y_actual_test = y_test
y_pred_test = lr.predict(X_test_standardized)

y_actual_train = y_train
y_pred_train = lr.predict(X_train_standardized)

# Compute accuracy for training and testing data
accuracy_test = accuracy_score(y_actual_test, y_pred_test)
accuracy_train = accuracy_score(y_actual_train, y_pred_train)

# Compute precision, recall, and f-score for training and testing data
precision_test = precision_score(y_actual_test, y_pred_test)
recall_test = recall_score(y_actual_test, y_pred_test)
f1_test = f1_score(y_actual_test, y_pred_test)

precision_train = precision_score(y_actual_train, y_pred_train)
recall_train = recall_score(y_actual_train, y_pred_train)
f1_train = f1_score(y_actual_train, y_pred_train)

print("================ Class Priors ================")
print("Class priors: ")
print("P(y=1): {:.3f}".format(np.mean(y_train)))
print("P(y=0): {:.3f}".format(1 - np.mean(y_train)))

print("================ Testing Data Performance  ================")
print("Accuracy for testing data: ", accuracy_test)
print("Precision for testing data: ", precision_test)
print("Recall for testing data: ", recall_test)
print("F1 Score for testing data: ", f1_test)

print("================ Training Data Performance  ================")
print("Accuracy for training data: ", accuracy_train)
print("Precision for training data: ", precision_train)
print("Recall for training data: ", recall_train)
print("F1 Score for training data: ", f1_train)


