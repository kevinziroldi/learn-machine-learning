# support vector machine o SVM
# è un classificatore binario, ovvero può dividere solo in due gruppi, che sono positivi e negativi
# consiste nel trovare il vettore che separa meglio i due gruppi
# il migliore vettore per separare è quello più distante possibile dai dati

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC # svc sta per support vector classifier - indichiamo il classificatore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

dataset = load_breast_cancer()

X = dataset['data']
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model_svc = SVC()
model_svc.fit(X_train, y_train)
prediction_svc = model_svc.predict(X_test)
acc_svc = accuracy_score(y_test, prediction_svc)
auc_svc = roc_auc_score(y_test, prediction_svc)
print('AUC svc:', auc_svc)
print('Acc svc:', acc_svc)

model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)
prediction_knn = model_knn.predict(X_test)
acc_knn = accuracy_score(y_test, prediction_knn)
auc_knn = roc_auc_score(y_test, prediction_knn)
print('AUC knn:', auc_knn)
print('Acc knn:', acc_knn)