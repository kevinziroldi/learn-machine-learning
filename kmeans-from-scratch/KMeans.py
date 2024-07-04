# k means è un algoritmo di unsupervised ML, ovvero non supervisionato: ovvero non si basa su dei dati di train di cui conosce già la classificazione
# quello che fa è suddividere i dati in gruppi anche detti 'clusters', successivamente quando c'è un nuovo dato vede a quale gruppo appartiene
# per trovare i gruppi io indico il valore di k, ovvero il numero di grouppi, lui prende inizialmente k punti a caso
# questi punti sono dei centri, calcola le distanze con gli altri e ogni punto apparterrà al gruppo del centro più vicino
# poi calcola il punto medio di tutti i gruppi e prende questi punti come nuovi centri, ogni volta sarà più preciso
# il processo va avanti fino a quando i punti non si muovono più, si muovono poco oppure l'algoritmo raggiunge il massimo numero di iterazioni che ho stabilto
# una volta trovai i centri effettivi, quando ho un nuovo dato non devo fare sempre il train, conosco i punti e calcolo la distanza solo per il nuovo punto 
# e stabilisco il gruppo di appartenenza, sempre con il criterio della distanza minore
# esistono poi due tipi di clustering: lineare (linear) o gerarchico (hierarchic)
# in quello lineare sono io a indicare il numero di centri e quindi di gruppi
# mentre in quello gerarchico è l'algoritmo stesso a capire quale possa essere il numero di gruppi da formare
# k means soffre di un problema ovvero i gruppi di diverse dimensioni, questo perchè si basa sulla distanza
# quindi tende a creare dei gruppi di dimensioni simili anche se ce ne sarebbero alcuni molto grandi e alcuni molto piccoli
# e questo genera per forza un errore di classificazione

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')

X = np.array([[1,2],
              [1.5,1.8],
              [5,8],
              [8,8],
              [1,0.6],
              [9,11]])

# plt.scatter(X[:,0], X[:,1], s=150, linewidths=5)
# plt.show()

clf = KMeans(n_clusters=2) # clf = classifier
clf.fit(X)

centroids = clf.cluster_centers_
print(centroids)
labels = clf.labels_
print(labels)

colors = 10*['g.', 'r.', 'c.', 'b.', 'k.']

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidths=5)
plt.show()