import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time

X_train = pd.read_csv('MLP_Xtrain_9x9.csv')
Y_train = pd.read_csv('MLP_Ytrain_9x9.csv')
Y_test = pd.read_csv('MLP_Ytest_9x9.csv')
X_test = pd.read_csv('MLP_Xtest_9x9.csv')

model_mlp = MLPRegressor(
        hidden_layer_sizes=(420, 350, 240, 160, 90),  activation='tanh', solver='lbfgs', alpha=0.0001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=50000, shuffle=True,
        random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
starttime_1 = time.time()
model_mlp.fit(X_train, Y_train)
endtime_1 = time.time()
totaltime_1 = endtime_1-starttime_1
model_knn = KNeighborsRegressor(n_neighbors=27, weights='distance')
starttime_2 = time.time()
model_knn.fit(X_train, Y_train)
endtime_2 = time.time()
totaltime_2 = endtime_2-starttime_2
model_rfr = RandomForestRegressor()
starttime_3 = time.time()
model_rfr.fit(X_train, Y_train)
endtime_3 = time.time()
totaltime_3 = endtime_3-starttime_3



#Y_pred_mlp = model_mlp.predict(X_test)
#print(model_mlp.score(X_train, Y_train))
#print(model_mlp.score(X_test, Y_test))
#print(mean_squared_error(Y_test, Y_pred_mlp))
print(totaltime_1)
print(totaltime_2)
print(totaltime_3)
joblib.dump(model_mlp, "MLP_model_9x9_inverse.m")
joblib.dump(model_knn, "KNN_model_9x9_inverse.m")
joblib.dump(model_rfr, "RF_model_9x9_inverse.m")
