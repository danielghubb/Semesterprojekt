import torch
import os
import pandas as pd
import numpy as np
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

class H5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.h5_file = h5py.File(file_path, 'r')
        self.group_names = list(self.h5_file.keys())

        # Initialize MinMaxScaler for normalization
        self.scaler = MinMaxScaler()

    def __len__(self):
        return len(self.group_names)

    def __getitem__(self, idx):
        group_name = self.group_names[idx]
        group = self.h5_file[group_name]
        
        # Assuming 'X' and 'Y' are datasets inside each group
        x_data = torch.Tensor(group['X'][:])
        y_data = torch.Tensor(group['Y'][:])

        # Flatten the data for MinMaxScaler
        x_data_flat = x_data.view(x_data.size(0), -1)

        # Normalize using MinMaxScaler
        x_data_normalized = torch.Tensor(self.scaler.fit_transform(x_data_flat.numpy()))

        # Reshape back to the original shape
        x_data_normalized = x_data_normalized.view(x_data.size())
        # Return only the first 7 variables
        return x_data_normalized[:7], y_data

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
