import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster (yData, method, ySize, noise):
#    match method:
#        case 1:
    #yData = yData.numpy()

    stelle = np.argmax(yData)
    zeile = ySize - np.floor(stelle / ySize)
    spalte = stelle % ySize

    yData = np.array(np.concatenate(yData, axis=0)).flatten()
    positives = np.array([i for i in range(len(yData)) if yData[i] > noise])

    if len(positives) < 2:
        return 0

    result_array = np.zeros((positives.shape[0], 2), dtype=int)
    result_array[:, 0] = ySize - np.floor(positives / ySize)  #y-wert
    result_array[:, 1] = positives % ySize                    #x-wert
        
    
        # KMeans-Modell erstellen und anpassen
    initial_x = (ySize/2)
    initial_centers = np.array([[ySize - np.floor(positives[0] / ySize), initial_x], [zeile, spalte], [ySize - np.floor(positives[-1] / ySize), initial_x]])

    kmeans = KMeans(n_clusters=3, init=initial_centers, n_init= 1)
    kmeans.fit(result_array)

            # Zugehörigkeit der Datenpunkte zu den Clustern abrufen
    labels = kmeans.labels_

    return silhouette_score(result_array, labels)




# Methode 1: clustering auf 2D Vektor mit k-means
