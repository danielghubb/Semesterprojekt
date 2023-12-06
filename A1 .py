import torch
import os
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import h5py

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torchvision import datasets
from torchvision import transforms

from sklearn.preprocessing import MinMaxScaler

##########Hyperparameter#############
batch_size = 32
##########Hyperparameter#############

#daten wurden durch script getMinMax generiert
mins = torch.Tensor(array([70.00147072, 1.50000067, 300.00161826, 0.00200653, 0.0014694, 3000, 0.5000061]))
maxs = torch.Tensor(array([149.99962444, 2.99992871, 799.99556485, 99.9966234, 99.9995793, 10000, 2.99250181]))

def normalize(tensor):
    for i in range (7):
        tensor[i] = (tensor[i]- mins[i])/(maxs[i]-mins[i])
    return tensor

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
        
        # Assuming 'X' and 'Y' are datasets inside each group
        x_data = torch.Tensor(group['X'][:])
        y_data = torch.Tensor(group['Y'][:])

        x_data_normal = normalize(x_data[:7])

        return x_data_normal, y_data

    def close(self):
        self.h5_file.close()

    def getmms(self):
        return self.scaler

# initialise dataset and dataloader
file_path = '/home/kali/Projects/Semesterprojekt/rzp-1_sphere1mm_train_100k.h5'

dataset = H5Dataset(file_path)

train, test =  torch.utils.data.random_split(dataset, [0.66, 0.34])
dataloader_train = DataLoader(train, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(test, batch_size=batch_size, shuffle=False)

# Accessing the first group in the dataset
sample_x, sample_y = dataset[0]
print("X data:", sample_x)
print("Y data:", sample_y)

# Closing the dataset
dataset.close()
