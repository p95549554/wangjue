import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import csv
import time




model_mlp = joblib.load("MLP_model_9x9.m")
Y_test = pd.read_csv('test_random.csv')
starttime = time.time()
Y_pred = model_mlp.predict(Y_test)
endtime = time.time()
totaltime = endtime-starttime
print(Y_test)
print(Y_pred)
print(totaltime)
f = open('X_pred_direct.csv', 'w', newline='')
csv_writer = csv.writer(f)
for i in Y_pred:
    csv_writer.writerow(i)
f.close()
