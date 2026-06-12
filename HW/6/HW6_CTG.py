import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB

# CTG Dataset

csvFilePath = './CTG.csv'

# Read in Data
df = pd.read_csv(csvFilePath)

df = df.drop(df.index[0])
df = df.drop(columns='CLASS')

feature_cols = ['LB','AC','FM','UC','ASTV','MSTV','ALTV','MLTV','DL','DS','DP','Width','Min','Max','Nmax','Nzeros','Mode','Mean','Median',
                'Variance','Tendency']
X = df[feature_cols].to_numpy()
Y = df['NSP'].to_numpy()

# Split Data Up
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=2/3, random_state=42, stratify=Y)

# Train Naive Bayes Classifier

classifier = GaussianNB(var_smoothing=0.1)
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)

# Compute class priors, accuracy, and generate confusion matrix
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