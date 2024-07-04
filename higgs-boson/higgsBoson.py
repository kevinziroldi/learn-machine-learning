import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
import pickle

df_train = pd.read_csv('training.csv')
df_test = pd.read_csv('test.csv')

rows_to_delete = []
for i in range(len(df_train)):
    for j in range(len(df_train.columns)):
        if df_train.iloc[i, j] == -999:
            rows_to_delete.append(i)
df_train.drop(rows_to_delete, axis=0, inplace=True)
            
rows_to_delete= []
for i in range(len(df_test)):
    for j in range(len(df_test.columns)):
        if df_test.iloc[i, j] == -999:
            rows_to_delete.append(i)
df_test.drop(rows_to_delete, axis=0, inplace=True)

y_train = df_train['Label']
columns_to_drop = ['Label', 'EventId', 'Weight']
df_train.drop(columns_to_drop, axis=1, inplace=True)
X_train = df_train

df_test.drop(['EventId'], axis=1, inplace=True)
X_test = df_test

columns = X_train.columns

transformers = [
    ['scaler', RobustScaler(), columns]
]
ct = ColumnTransformer(transformers, remainder='passthrough')
ct.fit_transform(X_train)
ct.fit_transform(X_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# usiamo pickle per salvare il modello e non doverlo addestrare ogni volta
with open('higgsBoson.pickle', 'wb') as f:
    pickle.dump(model, f)
# creo la prima volta il file.pickle

# poi posso commentare la parte in cui lo creo e alleno, mi risparmia molto tempo quando lavoro con molti file
with open('higgsBoson.pickle', 'rb') as pickle_file:
    model = pickle.load(pickle_file)

prediction_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, prediction_train)
print('Accuracy Train:', accuracy_train)

perdiction_test = model.predict(X_test)
print(perdiction_test)