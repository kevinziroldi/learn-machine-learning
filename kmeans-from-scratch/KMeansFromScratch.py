import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1,2],
              [1.5,1.8],
              [5,8],
              [8,8],
              [1,0.6],
              [9,11]])

# plt.scatter(X[:,0], X[:,1], s=20, linewidths=5)
# plt.show()

colors = 10*['g', 'r', 'c', 'b', 'k']

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self, data):
        self.centroids = {}
        
        # assegnamo come primi due centri i primi due dati, volendo avrei potuto scegliere random
        for i in range(self.k):
            self.centroids[i] = data[i]
            
        for i in range(self.max_iter):
            self.classification = {}    # classification è un dizionario, sarà formato da 'centro':[dati che fanno parte della classe di quel centroide]
                                        # in questo caso abbiamo k=2, quindi sarà {'centro 1':[dati classe 1], 'centro 2',[dati classe 2]}
            # creiamo i due elementi del dizionario                            
            for i in range(self.k):
                self.classification[i] = []
            
            # popoliamo con i dati il dizionario, dividendoli per classe  
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids] # creo una lista fatta da due elementi 
                                                                                                                 # che sono la distanza dal primo e dal secondo 
                                                                                                                 # centroide e, dato che sono in un for lo faccio 
                                                                                                                 # per tutti i dati di 'data'
                classification = distances.index(min(distances))
                self.classification[classification].append(featureset)
                
            prev_centroids = dict(self.centroids)
            
            for classification in self.classification:
                pass
                self.centroids[classification] = np.average(self.classification[classification], axis=0)

            optimized = True
            
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/(original_centroid*100.0)) > self.tol:
                    optimized = False # se il movimento del centro è maggiore della tolleranza allora non è ottimizzato
            
            if optimized:
                break
                
    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker='o', color='k', s=150, linewidths=5)
for classification in clf.classification:
    color = colors[classification]
    for featureset in clf.classification[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)
        
unknowns = np.array([[1,3],
                    [8,9],
                    [0,3],
                    [5,4],
                    [6,4]])
for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker='*', color=colors[classification], s=150, linewidths=5)
        
plt.show()