import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# taking the last day in input
print('Day before the one to predict')
d = int(input('day: '))
m = int(input('month: '))
y = int(input('year: '))

'''
y = 2020
m = 5
d = 4
'''

prediction_days = int(input('number of day for the predictions: '))
#prediction_days = 10
days_to_import = prediction_days + 100

last = datetime.date(y, m, d)

# find the starting day by subtracting 200 days
start = last - datetime.timedelta(days=days_to_import)

# importing the data
dataset = web.DataReader('AAPL', data_source='yahoo', start=start, end=last)

dataset = dataset['Close']

# plot the data
plt.plot(dataset)
plt.title('Close stock price graph')
plt.xlabel('Date', labelpad=10)
plt.ylabel('Close stock price', labelpad=10)
#plt.show()

X_train = []
y_train = []
for i in range(prediction_days, len(dataset)):
    X_train.append(dataset[i-prediction_days:i])
    y_train.append(dataset[i])

X_train = np.array(X_train, dtype=np.float64)
y_train = np.array(y_train, dtype=np.float64)

model = LinearRegression(n_jobs=10)
model.fit(X_train, y_train)

date_test = last + datetime.timedelta(days=1)
X_test = dataset[i-prediction_days:i]
X_test = np.array(X_test, dtype=np.float64)
X_test = X_test.reshape(1,-1)

prediction_test = model.predict(X_test)
#y_test = web.DataReader('AAPL', data_source='yahoo', start=date_test, end=date_test)
#y_test = np.array(y_test['Close'])
print('Predicted close price', date_test, ':', prediction_test)
#print('Actual close price', date_test, ':', y_test)