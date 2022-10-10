#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:54:10 2022

@author: ahmetkaansever
"""
#%% imports
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime as dt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import lightgbm as lgb
from pmdarima.arima import auto_arima
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVR
#%%
#Loading Dataset
data = pd.read_csv(r'/Users/ahmetkaansever/Desktop/Staj Proje/sunspot/sunspot.year.csv')
data = data[['time', 'value']]
a = data.head()
b = data.describe()
#data.hist()

#%%
#K_Means with 3 clusters
k = 3
kmeans = KMeans(n_clusters=k)
data['cluster'] = kmeans.fit_predict(data)

#data.plot(kind='scatter', x='time', y='value')

#Getting centroids
centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]

#%%
#Plotting K_Means Clustering--
colors = ['#DF2020', '#81DF20', '#2095DF', '#FFFF00', '#FFC0CB', '#000000']
plt.figure()
data.plot(kind = 'line', x='time', y='value')

#Plotting for k = 3

data['cen_x'] = data.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
data['cen_y'] = data.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})

data['c'] = data.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})

#Plotting
plt.figure()
plt.scatter(data.time, data.value, c=data.c, alpha = 0.6, s=10)

plt.figure()
# Extract Figure and Axes instance
fig, ax = plt.subplots()

# Create a plot
ax.violinplot([data['value']])

# Add title
ax.set_title('Sunspot Numbers')
ax.set_xticklabels(["Country GDP",])
ax.set_xticklabels(["Sunspot Numbers",])
plt.show()

#%%
#Finding Correlations
datacor = data[['time', 'value']]
corr_matrix = datacor.corr()
print(corr_matrix)
print(corr_matrix['value'].sort_values(ascending=False))

#%%
#Seasonal Decomposition
data['value'] = data['value'].replace(0,1)

plt.figure()
result=seasonal_decompose(data['value'], model='multiplicable', period=11)
result.seasonal.plot()
plt.title("Seasonal")
plt.savefig("Seasonal")
seasonal = result.seasonal

plt.figure()
result.trend.plot()
plt.title("Trend")
plt.savefig("trend")
trend = result.trend

plt.figure()
result.plot()

#%%##Converting Data to DateTime object
# =============================================================================
# #Converting time to DateTime object
# for i in  range(len(data['time'])):
#     data['time'][i] = str(data['time'][i]) + ('-01-01')
# 
# data['time'] = pd.to_datetime(data['time'])
# data.set_index('time', inplace=True)
# =============================================================================
#%%
#Checking if time series is stationary or not
def check_stationarity(ts):
    dftest = adfuller(ts)
    adf = dftest[0]
    pvalue = dftest[1]
    critical_value = dftest[4]['5%']
    if (pvalue < 0.05) and (adf < critical_value):
        print('The series is stationary')
    else:
        print('The series is NOT stationary')
        
check_stationarity(data['value'])
#Differantiating Data
ts_diff = data['value'].diff()
ts_diff.dropna(inplace=True)
check_stationarity(ts_diff)

dftest = adfuller(ts_diff, autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)

plt.figure()
plt.plot(ts_diff)
plt.title('Differenced Time Series')
plt.grid()
plt.show()

#ACF graphs of data to detect seasonality
plt.figure()
plot_acf(data['value'], lags =288)
plt.title("Autocorrelation 100 days")
plt.show()

plt.figure()
plot_acf(data['value'], lags =55)
plt.title("Autocorrelation 11 days")
plt.show()

plt.figure()
plot_pacf(data['value'], lags =100)
plt.title('Partialautocorrelation 100 days')
plt.show()

plt.figure()
plot_pacf(data['value'], lags =55)
plt.title('Partialautocorrelation 11 days')
plt.show()

# =============================================================================
# q=6
# 
# #Setting (P,D,Q,M)
# #Setting D
# check_stationarity(seasonal)
# D = 0
# 
# #Setting P
# 
# plt.figure()
# plot_pacf(data['value'], lags =50)
# plt.ylim(top = 1.5)
# plt.show()
# 
# P = 2
# 
# plot_acf(seasonal, lags =50)
# plt.figure()
# plt.show()
# 
# Q = 4
# =============================================================================

# =============================================================================
# model_seasonal = SARIMAX(ts_diff, order=(p,d,q), seasonal_order=(P,D,Q,50))
# model_fit_seasonal = model_seasonal.fit()
# print(model_fit_seasonal)
# 
# plt.figure()
# =============================================================================

#%% SARIMAX MODEL
# =============================================================================
# model= SARIMAX(data['value'], order=(3,1,2), seasonal_order=(2,1,2,11))
# results =model.fit()
# x=results.predict(start=250,end=300,dynamic=True)
# data['forecast']=results.predict(start=250,end=289,dynamic=True)
# data[['value','forecast']].plot(figsize=(12,8))
# 
# =============================================================================

#%% ARIMA MODEL
# =============================================================================
# model = ARIMA(data['value'], order=(3,1,3))
# results = model.fit()
# x=results.predict(start=250,end=300,dynamic=True)
# data['forecast']=results.redict(start=200,end=289,dynamic=True)
# plt.figure()
# data[['value','forecast']].plot(figsize=(12,8))
# =============================================================================

#%% SVR Regression
regr = SVR(kernel = 'poly', degree=6)
X = data['time']
y = data['value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, shuffle=False)
regr.fit(X.to_frame(), y)

polypred = regr.predict(X_test.to_frame())

plt.figure()
plt.plot(X_test, polypred, label='Prediction')
plt.plot(X_test, y_test, label='Actual')
#%% LightGBM Model and Feature Engineering
#Feature Engineering

lag1 = data['value'].shift(1).to_frame()
lag2 = data['value'].shift(2).to_frame()
lag3 = data['value'].shift(3).to_frame()
lag4 = data['value'].shift(4).to_frame()
lag5 = data['value'].shift(5).to_frame()
lag6 = data['value'].shift(6).to_frame()
lag7 = data['value'].shift(7).to_frame()
lag8 = data['value'].shift(8).to_frame()
lag9 = data['value'].shift(9).to_frame()
lag10 = data['value'].shift(10).to_frame()
lag11 = data['value'].shift(11).to_frame()


coord = ((data['value'].index % 11).to_frame())
coord.reset_index(drop=True, inplace=True)


value = data['value']

mean3 = lag1.rolling(window = 3).mean()
mean11 = lag1.rolling(window = 11).mean()
mean5 = lag1.rolling(window = 5).mean()
mean6 = lag1.rolling(window = 5).mean()
mean5_6 = value.shift(6).rolling(window=6).mean().to_frame()
mean3_5 = value.shift(3).rolling(window = 5).mean().to_frame()

sum1_6lag = lag1 - lag6
sum1_5lag = lag1 - lag5
sum1_4lag = lag1 + lag4
sum5_6mean = mean5_6 - mean5
sum1_11lag = lag11 - lag1
div1_11lag = lag11/lag1

extractedData = pd.concat([lag1, lag3, lag6, mean6, mean11,  mean5_6, div1_11lag, value], axis = 1)


extractedData = extractedData.iloc[11:,:]

extractedData.columns = ['lag1', 'lag3', 'lag6', 'mean6', 'mean11', 'mean5_6', 'div1_11lag', 'value']

X = extractedData.drop('value', axis = 1)
y = extractedData['value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, shuffle=(False))

lgbm = lgb.LGBMRegressor(n_jobs = 1, num_leaves = 10, max_depth = 4)
lgbm.fit(X_train, y_train)
print(lgbm)

predicted = lgbm.predict(X_test)
predicted = pd.DataFrame(predicted , columns = ['predicted'])
 
y_test = y_test.to_frame()
y_test.reset_index(drop=True, inplace=True)

graph = pd.concat([predicted, y_test], axis = 1)
print(graph)
print()
print(lgbm.feature_importances_)
print()

x1 = predicted
x2 = y_test

plt.figure()
plt.plot(x1)
plt.title('Predicted')
plt.figure()
plt.plot(x2)
plt.title('Actual')


plt.figure()
plt.plot(x1, label='predicted')
plt.plot(x2, label= 'actual')
plt.legend()
plt.title('LGBM')

print('R2 Score: ' + str(metrics.r2_score(y_test, predicted)))
print(metrics.mean_squared_log_error(y_test, predicted))

#%% Finding Correlations from extracted data

datacor = extractedData
corr_matrix = datacor.corr()
print(corr_matrix)
print(corr_matrix['value'].sort_values(ascending=False))


#%% ACCURACY
def forecast_accuracy(forecast, actual):

    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE

    me = np.mean(forecast - actual)             # ME

    mae = np.mean(np.abs(forecast - actual))    # MAE

    mpe = np.mean((forecast - actual)/actual)   # MPE

    return({'mape':mape, 'me':me, 'mae': mae, 

            'mpe': mpe})


accuracy = forecast_accuracy(predicted.to_numpy(), y_test.to_numpy())
print(accuracy)
