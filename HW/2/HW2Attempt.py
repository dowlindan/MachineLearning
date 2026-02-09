import os, sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import numpy as np

def RMSE(Y, Yhat):
    MSE = np.mean((Y-Yhat)**2)
    return MSE**(1/2)

def SMAPE(Y, Yhat):
    return np.mean((np.abs(Y - Yhat) / (np.abs(Y) + (np.abs(Yhat)))))

csvFilePath = './insurance.csv'

df = pd.read_csv(csvFilePath)

# Binary Encode Sex
df['sex'] = (df['sex'] == 'male').astype(int)

# Binary Encode Smoker
df['smoker'] = (df['smoker'] == 'yes').astype(int)

# OneHotEncode region 
ohe = OneHotEncoder(sparse_output=False, dtype=int)
region_ohe = ohe.fit_transform(df[['region']])
region_cols = [f"region_{cat}" for cat in ohe.categories_[0]]
region_df = pd.DataFrame(region_ohe, columns=region_cols, index=df.index)

# Drop original region column and add one-hot columns
df = pd.concat([df.drop('region', axis=1), region_df], axis=1)

print("Data: ")
print(df.head())

# Extract requested columns: features and target
feature_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']
X = df[feature_cols].to_numpy()
y = df['charges'].to_numpy()

# Split: 2/3 train, 1/3 test with a random seed of 0
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2/3, random_state=0)

# Linear Regression Model
lr = LinearRegression(fit_intercept=True)  # fit_intercept=True ensures explicit bias feature of 1
lr.fit(X_train, y_train)
print("Coef: ", lr.coef_)
print("Intercept: ", lr.intercept_)

# Predictions
y_actual_test = y_test
y_prediction_test = lr.predict(X_test)

y_actual_train = y_train
y_prediction_train = lr.predict(X_train)

# Print outcomes
rmse_test = RMSE(Y=y_actual_test, Yhat=y_prediction_test)
smape_test = SMAPE(Y=y_actual_test, Yhat=y_prediction_test)

rmse_train = RMSE(Y=y_actual_train, Yhat=y_prediction_train)
smape_train = SMAPE(Y=y_actual_train, Yhat=y_prediction_train)

print("RMSE for testing data: ", rmse_test)
print("SMAPE for testing data: ", smape_test)

print("RMSE for training data: ", rmse_train)
print("SMAPE for training data: ", smape_train)
