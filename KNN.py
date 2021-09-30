import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor

figure_size = []


x = pd.read_csv('MLP_Xtrain_9x9.csv')
y = pd.read_csv('MLP_Ytrain_9x9.csv')
#X_train = X_train.iloc[0:1700]
#Y_train = Y_train.iloc[0:1700]
#X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.05)
Y_test = pd.read_csv('MLP_Ytest_9x9.csv')
X_test = pd.read_csv('MLP_Xtest_9x9.csv')

print(Y_test)
for i in range(1, 11):
    data_size = i*300
    X_train = x.iloc[0:data_size-1]
    Y_train = y.iloc[0:data_size-1]
    figure_size.append(data_size)
    clf = KNeighborsRegressor(n_neighbors=57, weights='distance')
    rfr = RandomForestRegressor()
    clf.fit(X_train, Y_train)
    rfr.fit(X_train, Y_train)
    y_pred_knn = clf.predict(X_test)
    y_pred_rfr = rfr.predict(X_test)
#print(y_pred)
    print(data_size)
    print(clf.score(X_train, Y_train))
    print(clf.score(X_test, Y_test))
    print(mean_squared_error(Y_test, y_pred_knn))
    print(rfr.score(X_train, Y_train))
    print(rfr.score(X_test, Y_test))
    print(mean_squared_error(Y_test, y_pred_rfr))
#print(accuracy_score(y_pred,Y_test))