from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dataset = load_breast_cancer()

X = dataset['data']
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

prediction_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, prediction_train)

prediction_test = model.predict(X_test)
accuracy_test = accuracy_score(y_test, prediction_test)

print('Accuracy train:', accuracy_train)
print('Accuracy test:', accuracy_test)