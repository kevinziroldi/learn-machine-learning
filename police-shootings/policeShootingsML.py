import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

df = pd.read_csv('fatal-police-shootings-data.csv')

df.drop(['id'], 1, inplace=True)
df.dropna(inplace = True)

'''
ok = True
for i in range(len(df)):
    for col in df.columns:
        if df.loc[i, col] == np.nan or df.loc[i, col] == 'NaN' or df.loc[i, col] == 'nan':
            print(df.loc[i, col])
            ok = False
print(ok)
'''

# print(df.head())

X = pd.DataFrame()
X['armed'] = df.armed
X['age'] = df.age
X['gender'] = df.gender
X['signs_of_mental_illness'] = df.signs_of_mental_illness
X['threat_level'] = df.threat_level
X['flee'] = df.flee
# print(X)
transformers_X = [
    ['onehot', OneHotEncoder(), X.columns]
]
ct_X = ColumnTransformer(transformers_X, remainder='passthrough')
X = ct_X.fit_transform(X)

y = pd.DataFrame()
y['manner of death'] = df.manner_of_death
print(y)
transformers_y = [
    ['onehot', OneHotEncoder(), y.columns]
]
ct_y = ColumnTransformer(transformers_y, remainder='passthrough')
ct_y.fit(y)
y = ct_y.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

svm = SVC(C=0.0001)
svm.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

prediction_test_svm = svm.predict(X_test)
accuracy_test_svm = accuracy_score(y_test, prediction_test_svm)
# print('Accuracy test svm:', accuracy_test_svm)

prediction_test_knn = knn.predict(X_test)
accuracy_test_knn = accuracy_score(y_test, prediction_test_knn)
# print('Accuracy test knn:', accuracy_test_knn)

svm_auc = roc_auc_score(y_test, accuracy_test_svm)
knn_auc = roc_auc_score(y_test, accuracy_test_knn)