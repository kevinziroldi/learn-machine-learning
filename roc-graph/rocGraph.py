from sklearn.datasets import make_classification
X, Y = make_classification(n_samples=2000, n_classes=2, n_features=10, random_state=0)

# complico un po' il dataset per renderlo più relaistico
import numpy as np
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# divido i dati in train e test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=0)

# creiamo due modelli di classificazione
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# 1) random forest
rf = RandomForestClassifier(max_features=5, n_estimators=500)
rf.fit(X_train, Y_train)

# 2) naive bayes
nb = GaussianNB()
nb.fit(X_train, Y_train)
 
# prediction probabilities
r_probs = [0 for _ in range(len(Y_test))]
rf_probs = rf.predict_proba(X_test)
nb_probs = nb.predict_proba(X_test)

rf_probs = rf_probs[:, 1]
nb_probs = nb_probs[:, 1]

# calculate AUC
from sklearn.metrics import roc_curve, roc_auc_score
r_auc =  roc_auc_score(Y_test, r_probs) # caso peggiore, in cui tutte le predizioni sono sbagliate
print(Y_test)
print(rf_probs)
print(nb_probs)
rf_auc = roc_auc_score(Y_test, rf_probs)
nb_auc = roc_auc_score(Y_test, nb_probs)
print('Random (chance) prediction, AUC =', r_auc)
print('Random forest, AUC =', rf_auc)
print('Naive bayes, AUC = ', nb_auc)

# calculare ROC
r_fpr, r_tpr, _ = roc_curve(Y_test, r_probs)
rf_fpr, rf_tpr, _ = roc_curve(Y_test, rf_probs)
nb_fpr, nb_tpr, _ = roc_curve(Y_test, nb_probs)
# draw ROC graph
import matplotlib.pyplot as plt
plt.plot(r_fpr, r_tpr, linestyle='--', label='Randon Prediction (AUC =' + str(r_auc) + ')')
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUC=' + str(rf_auc) + ')')
plt.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes (AUC=' + str(nb_auc) + ')')
plt.title('ROC graphs')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()