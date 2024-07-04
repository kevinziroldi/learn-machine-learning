import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('bmh')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# voglio predire la temperatura percepita basandomi su altre caratteristiche

df = pd.read_csv('weatherHistory.csv')

to_drop = [i for i in range(1000, len(df))]
df.drop(to_drop, axis=0, inplace=True)

y = df['Apparent Temperature (C)']

df.drop(['Formatted Date', 'Summary', 'Precip Type', 'Apparent Temperature (C)', 'Visibility (km)', 'Loud Cover', 'Pressure (millibars)', 'Daily Summary'],
        axis=1, inplace=True)

# senza considerare il vento, ma è un modello peggiore
df.drop(['Formatted Date', 'Summary', 'Precip Type', 'Apparent Temperature (C)', 'Visibility (km)', 'Loud Cover', 'Pressure (millibars)', 'Daily Summary', 'Wind Speed (km/h)', 'Wind Bearing (degrees)'],
        axis=1, inplace=True)

for feature in df.columns:
    for i in range(len(df)):
        plt.scatter(df.loc[i, feature], y[i])
    plt.xlabel(feature)
    plt.ylabel('Apparent Temperature (C)')
    plt.show()

transformers = [
    ['scaler', RobustScaler(), ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)']]
]

# senza considerare il vento, ma è un modello peggiore
transformers = [
    ['scaler', RobustScaler(), ['Temperature (C)', 'Humidity']]
]

ct = ColumnTransformer(transformers, remainder='passthrough')
X = ct.fit_transform(df)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model1 = LinearRegression(n_jobs=10)
model1.fit(X_train, y_train)

model2 = SVR()
model2.fit(X_train, y_train)

with open('weatherHistory1.pickle', 'wb') as f:
    pickle.dump(model1, f)
with open('weatherHistory2.pickle', 'wb') as f:
    pickle.dump(model2, f)

#with open('weatherHistory1.pickle', 'rb') as pickle_file:
#    model1 = pickle.load(pickle_file)
with open('weatherHistory2.pickle', 'rb') as pickle_file:
    model2 = pickle.load(pickle_file)

#prediction_train1 = model1.predict(X_train)
#prediction_test1 = model1.predict(X_test)
prediction_train2 = model2.predict(X_train)
prediction_test2 = model2.predict(X_test)

#mae_train1 = mean_absolute_error(y_train, prediction_train1)
#mae_test1 = mean_absolute_error(y_test, prediction_test1)
mae_train2 = mean_absolute_error(y_train, prediction_train2)
mae_test2 = mean_absolute_error(y_test, prediction_test2)

#mse_train1 = mean_squared_error(y_train, prediction_train1)
#mse_test1 = mean_squared_error(y_test, prediction_test1)
mse_train2 = mean_squared_error(y_train, prediction_train2)
mse_test2 = mean_squared_error(y_test, prediction_test2)

#r2_train1 = r2_score(y_train, prediction_train1)
#r2_test1 = r2_score(y_test, prediction_test1)
r2_train2 = r2_score(y_train, prediction_train2)
r2_test2 = r2_score(y_test, prediction_test2)

#print('MAE Train linear regression:', mae_train1)
#print('MAE Test linear regression:', mae_test1)
#print('MSE Train linear regression:', mse_train1)
#print('MSE Test linear regression:', mse_test1)
#print('R2 Train linear regression:', r2_train1)
#print('R2 Test linear regression:', r2_test1)
#print()
print('MAE Train SVM regressor:', mae_train2)
print('MAE Test SVM regressor:', mae_test2)
print('MSE Train SVM regressor:', mse_train2)
print('MSE Test SVM regressor:', mse_test2)
print('R2 Train SVM regressor:', r2_train2)
print('R2 Test SVM regressor:', r2_test2)

# mae migliore SVR, mse migliore per LinearRegression, r2 pressochè uguale, leggermente meglio per SVR

y_test = np.array(y_test).reshape(-1,1)
for feature in range(len(df.columns)):
    for i in range(len(X_test)):
        plt.scatter(X_test[i][feature], y_test[i], c='k')
        plt.scatter(X_test[i][feature], prediction_test[i], c='r')
    plt.xlabel(df.columns[feature])
    plt.ylabel('Apparent Temperature (C)')
    plt.show()
