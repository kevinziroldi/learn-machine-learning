import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

dataset_train = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dataset_test = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

y_train = dataset_train.pop('survived')
X_train = dataset_train
y_test = dataset_test.pop('survived')
X_test = dataset_test

categorical_columns = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
numeric_columns = ['age', 'fare']

# ogni volta c'è una cosa da fare: creare le feature columns, che sono un encoding: io posso passare al modello solo dati numerici, in tensorflow ci sono
# delle funzioni che mi permettono di rendere numerici dei dati categorici in automatico
# queste funzioni rispettano delle particolari sintassi, qui un esempio da seguire ogni volta che dovrò fare questo tipo di problemi

feature_columns = []

for feature_name in categorical_columns:
    vocabulary = X_train[feature_name].unique() # serve per prendere tutti i valori delle colonne di categorie (ad esempio le città o maschio e femmina), non si ripetono se sono presenti più di una volta
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)) # funzione di tensorflow
    
for feature_name in numeric_columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name)) # funzione di tensorflow, essendo numerico NON serve prima la parte con .unique

print(feature_columns)


# i modelli di tensorflow lavorano solo con degli oggetti specifici: tf.data.Dataset
# per fare questo si usa questa funzione, presa direttamente dalla documentazione di tensorflow

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():  # inner function, this will be returned
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
        return ds  # return a batch of the dataset
    return input_function  # return a function object for use
    
train_input_fn = make_input_fn(X_train, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(X_test, y_test, num_epochs=1, shuffle=False)


# creiamo il modello vero e proprio

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)


# alleniamo il modello

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

print(result['accuracy'])  # the result variable is simply a dict of stats about our model


# predict values

pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')