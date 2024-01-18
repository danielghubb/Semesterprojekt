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
from sklearn.mixture import GaussianMixture


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
noise = 0.0000 #sehr empfindlich!
dataset = H5Dataset(file_path)



def cluster (yData, method):
    match method:

        case 1:
            yData = yData.numpy()
            yData = np.array(np.concatenate(yData, axis=0)).flatten()
            positives = np.array([i for i in range(len(yData)) if yData[i] > noise])

            result_array = np.zeros((positives.shape[0], 2), dtype=int)
            result_array[:, 0] = ySize - np.floor(positives / ySize)  #y-wert
            result_array[:, 1] = positives % ySize                    #x-wert
            
            # KMeans-Modell erstellen und anpassen
            initial_x = (ySize/2)
            initial_centers = np.array([[0, initial_x], [initial_x, initial_x], [ySize, initial_x]])
            #initial_centers = np.array([[initial_x, 0], [initial_x, initial_x], [initial_x, ySize]])
            kmeans = KMeans(n_clusters=3, init=initial_centers, n_init= 1)
            kmeans.fit(result_array)

            # Zugehörigkeit der Datenpunkte zu den Clustern abrufen
            labels = kmeans.labels_

            # Zentren der Cluster
            centers = kmeans.cluster_centers_

            # Create a scatter plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(y)
            axes[0].set_title('Plot 1: real Data')

            axes[1].scatter(result_array[:, 1], result_array[:, 0], marker='o', c=labels, label='Scatter Plot')
            #axes[1].scatter(initial_centers[:,1], initial_centers[:,0], c='red', marker='X', s=200, label='Cluster Centers')
            axes[1].scatter(centers[:,1], centers[:,0], c='black', marker='X', s=200, label='Cluster Centers')
            axes[1].scatter(initial_centers[:,1], initial_centers[:,0], c='black', marker='X', s=1, label='Cluster Centers')
            axes[1].scatter(initial_centers[:,0], initial_centers[:,1], c='black', marker='X', s=1, label='Cluster Centers')
            axes[1].set_title('Plot 2: Clustering')

            # Set the same size for both subplots
            axes[0].set_aspect('equal', adjustable='datalim')
            axes[1].set_aspect('equal', adjustable='datalim')

            # Adjust layout
            plt.tight_layout()

            # Show the plots
            plt.show()
        

        case 2:
            yData = yData.numpy()
            yData = np.array(np.concatenate(yData, axis=0)).flatten()
            positives = np.array([i for i in range(len(yData)) if yData[i] > 0])
            positives_numbers = np.array([yData[i ]for i in range(len(yData)) if yData[i] > noise])*20  ###20 ist ein random wert um einfluss von epsilon zu beschränken, maybe zurück normalisieren
            result_array = np.zeros((positives.shape[0], 2), dtype=int)
            result_array[:, 0] = ySize - np.floor(positives / ySize)    #y-Wert
            result_array[:, 1] = positives % ySize                      #x-Wert
            
            # DBSCAN-Modell erstellen und anpassen
            dbscan = DBSCAN(eps=5, min_samples=1)
            labels = dbscan.fit_predict(result_array, sample_weight = positives_numbers)

            # Create a scatter plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(y)
            axes[0].set_title('Plot 1: real Data')

            unique_labels = np.unique(labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

            #print(len(set(labels)) - (1 if -1 in labels else 0))  #nachher vlt zum loss berechnen sinnvoll

            for label, color in zip(unique_labels, colors):
                cluster_points = result_array[labels == label]
                if label != -1: #damit noise nicht geplottet wird
                    axes[1].scatter(cluster_points[:, 1], cluster_points[:, 0], c=[color], label=f'Cluster {label}')
            axes[1].set_title('Plot 2: Clustering')

            span = np.array([[0, 0], [ySize, ySize]])
            axes[1].scatter(span[:,0], span[:,1], c='black', marker='X', s=1, label='Cluster Centers')
            # Set the same size for both subplots
            axes[0].set_aspect('equal', adjustable='datalim')
            axes[1].set_aspect('equal', adjustable='datalim')
            plt.tight_layout()

            # Show the plots
            plt.show()

        case 3:
            yData = yData.numpy()
            yData = np.array(np.concatenate(yData, axis=0)).flatten()
            positives = np.array([i for i in range(len(yData)) if yData[i] > noise])
            result_array = np.zeros((positives.shape[0], 2), dtype=int)
            result_array[:, 0] = ySize - np.floor(positives / ySize)  #y-wert
            result_array[:, 1] = positives % ySize                    #x-wert

            # KMeans-Modell erstellen und anpassen
            initial_x = (ySize/2)
            initial_centers = np.array([[0, initial_x], [initial_x, initial_x], [ySize, initial_x]])
            kmeans = KMeans(n_clusters=3, init=initial_centers, n_init= 1)
            kmeans_labels = kmeans.fit_predict(result_array)

            # Apply GMM clustering to the K-Means cluster centers
            gmm = GaussianMixture(n_components=3)
            gmm.fit(kmeans.cluster_centers_)
            gmm_labels = gmm.predict(result_array)

            # Create a scatter plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(y)
            axes[0].set_title('Plot 1: real Data')

            axes[1].scatter(result_array[:, 1], result_array[:, 0], marker='o', c=gmm_labels, label='Scatter Plot')
            span = np.array([[0, 0], [ySize, ySize]])
            axes[1].scatter(span[:,0], span[:,1], c='black', marker='X', s=1, label='Cluster Centers')
            axes[1].set_title('Plot 2: Clustering')

            # Set the same size for both subplots
            axes[0].set_aspect('equal', adjustable='datalim')
            axes[1].set_aspect('equal', adjustable='datalim')
            # Adjust layout
            plt.tight_layout()
            # Show the plots
            plt.show()

        case 4:
            yData = yData.numpy()
            yData = np.array(np.concatenate(yData, axis=0)).flatten()
            positives = np.array([i for i in range(len(yData)) if yData[i] > 0])
            positives_numbers = np.array([yData[i ]for i in range(len(yData)) if yData[i] > noise])*20

            result_array = np.zeros((positives.shape[0], 2), dtype=int)
            result_array[:, 0] = ySize - np.floor(positives / ySize)  #y-wert
            result_array[:, 1] = positives % ySize                    #x-wert

            # Apply GMM clustering to the K-Means cluster centers
            gmm = GaussianMixture(n_components=3)
            gmm.fit(result_array)
            gmm_labels = gmm.predict(result_array)

            # Create a scatter plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(y)
            axes[0].set_title('Plot 1: real Data')

            axes[1].scatter(result_array[:, 1], result_array[:, 0], marker='o', c=gmm_labels, label='Scatter Plot')
            span = np.array([[0, 0], [ySize, ySize]])
            axes[1].scatter(span[:,0], span[:,1], c='black', marker='X', s=1, label='Cluster Centers')
            axes[1].set_title('Plot 2: Clustering')

            # Set the same size for both subplots
            axes[0].set_aspect('equal', adjustable='datalim')
            axes[1].set_aspect('equal', adjustable='datalim')
            # Adjust layout
            plt.tight_layout()
            # Show the plots
            plt.show()



# Methode 1: clustering auf 2D Vektor mit k-means
# Methode 2: clustering auf 2D Vektor mit DBSCAN
# Methode 3: clustering auf 2D Vektor mit k-means plus Gaussian-Mixture-Model
# Methode 4: clustering auf 2D Vektor Gaussian-Mixture-Model


for i in range(3):
    sample = random.randrange(90000)
    x, y = dataset[sample]
    cluster(y,4)


dataset.close()
