# per fare l'algoritmo dobbiamo calcolare la distanza euclidea, radice di tutti i termini alla seconda sommati
# dipende dal numero di dimensioni dello spazio che considero

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
from math import sqrt

'''
plot1 = [1, 3]
plot2 = [2,5]
euclidean_distance = sqrt( (plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2)
print(euclidean_distance)
# in realtà esiste una funzione numpy che calcola la distanza euclidea
''' 

style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]



def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K ha un valore più basso del numero di elementi!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], color=result)
plt.show()