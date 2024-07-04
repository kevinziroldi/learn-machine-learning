import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.metrics import mean_squared_error

df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2020-01-01')
#print(df)

#print(df.shape)

# visualize the closing price history
'''
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=14, labelpad=15)
plt.ylabel('Close Price USD', fontsize=14, labelpad=15)
plt.show()
'''

# create a new dataframe with only close columns
data = df.filter(['Close'])

# convert the dataset to a numpy array
dataset = np.array(data)
# get the number of rows to train the model on
training_data_len = math.ceil(len(dataset)*0.8)
#print(training_data_len)

# scale the data
'''
transformers = [
    ['scaler', MinMaxScaler(feature_range=(0, 1)), list]
]
ct = ColumnTransformer(transformers, remainder='passthrough')
scaled_data = ct.fit_transform(data)
# convert the dataset to a numpy array
scaled_data = np.array(scaled_data)
'''
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# create the training dataset:
# scaled training dataset
train_data = scaled_data[:training_data_len, :]

# split in x_train and y_train
X_train = [] 
y_train = []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
# in pratica la predizione non si basa su altre feature, ma solo sui precedenti 60 valori

# convert X_train and y_train to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# reshape data
# devo cambiare la forma perchè l'algoritmo che usiamo, LSTM aspetta dati a 3 dimensioni: number of samples, number of time steps e number of feature 
#print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#print(X_train.shape)

# build the model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(train_data.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# test dataset
train_data = scaled_data[:training_data_len, :]
test_data = scaled_data[training_data_len-60:, :]

# split in X_test and y_test
X_test = []
y_test = dataset[training_data_len:, :] # sono tutti i dati che non abbiamo usato per il modello
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
    
# convert data to numpy arrays and reshape
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# get the model predicted price values
prediction_test = model.predict(X_test)
prediction_test = scaler.inverse_transform(prediction_test)

# measure how well the model predicted using root mean squared error
rmse = math.sqrt(mean_squared_error(y_test, prediction_test))
print('Root Mean Squared Error:', rmse)

# plot the data
train = data[0:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = prediction_test
plt.figure(figsize = (16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.plot(train['Close'])
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Prediction'], loc='lower right')
plt.show()

# show actual and predicted prices
print(validation)

apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2020-01-01')
new_df = apple_quote.filter(['Close'])
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

# fa un po' cagare perchè per ogni giorno si basa sui 60 precedenti, quindi non predice realmente alla distanza di mesi o anni come potrebbe sembrare