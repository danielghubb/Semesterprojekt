import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset, DataLoader
import random

from sklearn.cluster import KMeans

import hdbscan
import plotly.express as px

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs


class H5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.h5_file = h5py.File(file_path, 'r')
        self.group_names = list(self.h5_file.keys())

    def __len__(self):
        return len(self.group_names)

    def __getitem__(self, idx):
        group_name = self.group_names[idx]
        group = self.h5_file[group_name]
        
        x_data = torch.Tensor(group['X'][:])
        y_data = torch.Tensor(group['Y'][:])

        return x_data[:7], y_data

    def close(self):
        self.h5_file.close()

#Hyperparameter
file_path = '/home/kali/Projects/Semesterprojekt/normed_data.h5'
ySize = 256
dataset = H5Dataset(file_path)

# Accessing the first group in the dataset
#x, y = dataset[3333]
#x, y = dataset[33333]
#x, y = dataset[111]
#plt.imshow(y)
#plt.show()




def cluster (yData, method):
    match method:

        case 1:
            #histogram flatten
            sample_y = yData.numpy()
            histogram_data = np.array(np.concatenate(sample_y, axis=0))
            histogram_data_2d = histogram_data.reshape(-1, 1)

            # KMeans-Modell erstellen und anpassen
            kmeans = KMeans(n_clusters=3, random_state=42, n_init= 10)
            kmeans.fit(histogram_data_2d)

            # Zugehörigkeit der Datenpunkte zu den Clustern abrufen
            labels = kmeans.labels_

            # Zentren der Cluster
            centers = kmeans.cluster_centers_

            # Plot der Originaldaten und der Clusterzentren
            plt.scatter(range(len(histogram_data)), histogram_data, c=labels, cmap='viridis', edgecolor='k', s=40)
            plt.scatter(range(3), centers, c='red', marker='X', s=200, label='Cluster Centers')
            plt.title('k-means 1D')
            plt.xlabel('Datenpunkte')
            plt.ylabel('Histogrammwerte')
            plt.legend()
            plt.show()

        case 2:
            #histogramm flatten
            sample_y = yData.numpy()
            histogram_data = np.array(np.concatenate(sample_y, axis=0)).flatten()

            #hdbscan erstellen und anpassen
            clusterer = hdbscan.HDBSCAN(min_cluster_size=3,min_samples=8)
            clusterer.fit(histogram_data.reshape(-1, 1))

            # Zugehörigkeit der Datenpunkte zu den Clustern abrufen
            labels = clusterer.labels_

            plt.scatter(range(len(histogram_data)), histogram_data, c=labels, cmap='viridis', edgecolor='k', s=40)
            plt.title('HDBSCAN 1D')
            plt.xlabel('Datenpunkte')
            plt.ylabel('Histogrammwerte')
            plt.legend()
            plt.show()

        case 3:
            yData = yData.numpy()
            yData = np.array(np.concatenate(yData, axis=0)).flatten()
            positives = np.array([i for i in range(len(yData)) if yData[i] > 0])
            #print(positives)
            result_array = np.zeros((positives.shape[0], 2), dtype=int)
            result_array[:, 0] = np.floor(positives / ySize)
            result_array[:, 1] = positives % ySize
            #print(result_array)
            
            # KMeans-Modell erstellen und anpassen
            kmeans = KMeans(n_clusters=3, random_state=42, n_init= 10)
            kmeans.fit(result_array)

            # Zugehörigkeit der Datenpunkte zu den Clustern abrufen
            labels = kmeans.labels_

            # Zentren der Cluster
            centers = kmeans.cluster_centers_

            # Extract x and y values from the 2D array
            x_values = result_array[:, 0]
            y_values = result_array[:, 1]

            # Create a scatter plot
            plt.scatter(y_values, x_values, marker='o', c=labels, label='Scatter Plot')
            # Customize the plot
            plt.title('k-Means 2D')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.grid(True)

            # Show the plot
            plt.show()


        case 4:
            yData = yData.numpy()
            yData = np.array(np.concatenate(yData, axis=0)).flatten()
            positives = np.array([i for i in range(len(yData)) if yData[i] > 0])
            result_array = np.zeros((positives.shape[0], 2), dtype=int)
            result_array[:, 0] = np.floor(positives / ySize)
            result_array[:, 1] = positives % ySize
            
            # DBSCAN-Modell erstellen und anpassen
            dbscan = DBSCAN(eps=1.5, min_samples=3) #2,10
            labels = dbscan.fit_predict(result_array)

            # Plot the clusters
            unique_labels = np.unique(labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                cluster_points = result_array[labels == label]
                plt.scatter(cluster_points[:, 1], cluster_points[:, 0], c=[color], label=f'Cluster {label}')

            plt.title('DBSCAN  2D')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            plt.show()
        

        case 5:
            yData = yData.numpy()
            yData = np.array(np.concatenate(yData, axis=0)).flatten()
            positives = np.array([i for i in range(len(yData)) if yData[i] > 0])
            positives_numbers = np.array([yData[i ]for i in range(len(yData)) if yData[i] > 0])*20  ###20 ist ein random wert um einfluss von epsilon zu beschränken, maybe zurück normalisieren
            #print(positives_numbers)
            result_array = np.zeros((positives.shape[0], 2), dtype=int)
            result_array[:, 0] = ySize - np.floor(positives / ySize)    #y-Wert
            result_array[:, 1] = positives % ySize              #x-Wert
            
            # DBSCAN-Modell erstellen und anpassen
            dbscan = DBSCAN(eps=5, min_samples=1)
            labels = dbscan.fit_predict(result_array, sample_weight = positives_numbers)

            # Plot the clusters
            unique_labels = np.unique(labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

            print(len(set(labels)) - (1 if -1 in labels else 0))  #nachher vlt zum loss berechnen sinnvoll

            for label, color in zip(unique_labels, colors):
                cluster_points = result_array[labels == label]
                plt.scatter(cluster_points[:, 1], cluster_points[:, 0], c=[color], label=f'Cluster {label}')

            plt.title('weighted DBSCAN  2D')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            plt.show()

# Methode 1: clustering auf 1D Vektor mit k-means (fail)
# Methode 2: clustering auf 1D Vektor mit HDBSCAn (fail)
# Methode 3: clustering auf 2D Vektor mit k-means (fail)
# Methode 4: clustering auf 2D Vektor mit DBSCAN (gut)
# Methode 5  clustering auf 2D Vektor mit DBSCAN (träumchen)


#fig, axes = plt.subplots(2, 1, figsize=(12, 8))

for i in range(3):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plt.tight_layout()

    sample = random.randrange(90000)
    x, y = dataset[sample]
    axes[0].imshow(y)

    cluster(y,5)


dataset.close()











