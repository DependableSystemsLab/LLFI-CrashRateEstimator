import sys
import os
import numpy as np
import pandas as pd
import csv
from datetime import datetime

from sklearn import datasets
from sklearn.linear_model import BayesianRidge, Lasso, Ridge, LinearRegression, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, r2_score

try:
    data_input = sys.argv[1]
    model_filename = sys.argv[2]
except:
    print ("Please include input filenames.")
    print ("USAGE: python3 predict.py input_data.csv model.joblib")
    exit()

if not os.path.isfile(data_input):
    print ("File " + data_input + " not found.")
    print ("USAGE: python3 predict.py input_data.csv model.joblib")
    exit()

if not os.path.isfile(model_filename):
    print ("File " + model_filename + " not found.")
    print ("USAGE: python3 predict.py input_data.csv model.joblib")
    exit()

data = pd.read_csv(data_input).values #create numpy array from csv 
programs = data[:,0] # first column is the program name
X = data[:,1:4] # next 3 columns are X
n, d = X.shape

print ("Number of data samples = " + str(n))

# load model from file 
estimator = joblib.load(model_filename)
# make predictions and save to csv
y = estimator.predict(X)
results = np.column_stack((programs,y))
np.savetxt("predictions.csv", results, fmt="%10s,%10.3f")

print ("Saved crash rate estimates to predictions.csv")
