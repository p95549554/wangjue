import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import csv
import time




model_mlp = joblib.load("MLP_model_9x9_inverse.m")
model_knn = joblib.load("KNN_model_9x9_inverse.m")
model_rfr = joblib.load("RF_model_9x9_inverse.m")
X_test = pd.read_csv('test_random.csv')
starttime_1 = time.time()
Y_pred_mlp = model_mlp.predict(X_test)
endtime_1 = time.time()
totaltime_1 = endtime_1-starttime_1

starttime_2 = time.time()
Y_pred_knn = model_knn.predict(X_test)
endtime_2 = time.time()
totaltime_2 = endtime_2-starttime_2

starttime_3 = time.time()
Y_pred_rfr = model_rfr.predict(X_test)
endtime_3 = time.time()
totaltime_3 = endtime_2-starttime_3

print(X_test)
print(Y_pred_mlp)
print(Y_pred_knn)
print(Y_pred_rfr)
print(totaltime_1)
print(totaltime_2)
print(totaltime_3)
#f = open('Y_pred_direct.csv', 'w', newline='')
#csv_writer = csv.writer(f)
#for i in Y_pred:
#    csv_writer.writerow(i)
#f.close()
