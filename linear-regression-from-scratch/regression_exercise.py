# l'idea alla base della regressione è quella di trovare la linea che meglio si adatti ai dati che noi abbiamo
# esistono vari tipi di regressione che fanno utilizzo di varie linee
# la più semplice è la regressione lineare: quindi la linea sarà una retta. L'equazione di una retta è y = mx + q
# quindi dovrò identificare gli m e q che identificano la linea che descriva nel miglior modo possibile i miei dati

import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL', authtoken='UhYPVbaiErJ_CKYvhwoy')

# devo cercare di semplificarlo per rendere meno pesante l'algoritmo e tenere solo le caratteristiche utili
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

df['HL_Percent'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['Percent_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_Percent', 'Percent_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) # è meglio non eliminare i dati con null, perchè perderemmo molti dati, ma magari ne manca solo uno dei molti
# la miglior cosa da fare è sostituire, con quel valore perchè dopo si capirà

forecast_out = int(math.ceil(0.01 * len(df)))
# print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = LinearRegression(n_jobs=10) # se volessi cambiare algoritmo basterebbe sostituire questa riga di codice con clf = svm.SVR()    -    njobs serve per usare più potenza di calcolo e esegurie più velocemente il train
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

# print(accuracy)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

print(df)