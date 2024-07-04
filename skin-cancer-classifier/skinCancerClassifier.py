import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img)) # convert the images into arrays
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

def normalize(X):
    min_val = np.min(X)
    max_val = np.max(X)
    X = (X-min_val) / (max_val-min_val)
    return X


DATADIR = '/Users/kevin/Desktop/deepLearning/skinCancer/datasetSkinCancer'
CATEGORIES = ['benign', 'malignant']

training_data = []
IMG_SIZE = 224

'''
create_training_data()

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y)

X = normalize(X)

pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()
'''

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)

pickle_in = open('y.pickle', 'rb')
y = pickle.load(pickle_in)

# modello preso da: https://www.tensorflow.org/tutorials/images/classification
# ma ha accuracy train: 0.5029 e accuracy test: 0.4893, quindi cercare di meglio
num_classes = 2
model = keras.Sequential([
    keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, y)

model.fit(X_train, y_train, epochs=10)

loss_test, accuracy_test = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy_test)

# video: https://www.youtube.com/watch?v=AACPaoDsd50&t=755s