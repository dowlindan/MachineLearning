import os, sys
import pandas as pd
from sklearn.linear_model import LinearRegression

csvFilePath = './insurance.csv'
data = pd.read_csv(csvFilePath)


