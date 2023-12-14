import copy
from os import write
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import h5py

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from randomSplit import random_split


##########Hyperparameter#############
file_path = r'../Semesterprojekt/data.h5'
batch_size = 32
lr = 0.1
##########Hyperparameter#############

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)

#daten wurden durch script getMinMax generiert
mins = torch.Tensor(array([70.00147072, 1.50000067, 300.00161826, 0.00200653, 0.0014694, 3000, 0.5000061]))
maxs = torch.Tensor(array([149.99962444, 2.99992871, 799.99556485, 99.9966234, 99.9995793, 10000, 2.99250181]))

def normalize(tensor):
    for i in range (7):
        tensor[i] = (tensor[i]- mins[i])/(maxs[i]-mins[i])
    return tensor

def denormalize(tensor):
    for i in range(256):
        for j in range(256):
            tensor[i][j] = tensor[i][j] * (maxs[i] - mins[i]) + mins[i]
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
dataset = H5Dataset(file_path)

train, test =  random_split(dataset, [0.66, 0.34])
dataloader_train = DataLoader(train, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(test, batch_size=batch_size, shuffle=False)

# Accessing the first group in the dataset
#sample_x, sample_y = dataset[0]
#print(dataset[0][0])
#print("X data:", sample_x)
#print("Y data:", sample_y)

class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()

        # Input layer for 7 variables
        self.input_layer = nn.Linear(7, 1024)
        # Fully connected layer to connect to the deconvolution layers
        self.fc_layer = nn.Linear(1024, 256 * 32 * 32)  # Adjusted to match the product of deconv1 input size
        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # Adjusted kernel_size and output channels
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # Adjusted output channels and kernel_size
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)    # Adjusted output channels and kernel_size

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        # Fully connected layer
        x = self.fc_layer(x)
        x = x.reshape(-1, 256, 32, 32)  # Reshape to match deconv1 input size
        # Deconvolution layers
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = x.view(-1, 256, 256).squeeze()

        return x
    
model = torch.load('./model_5.pth')
for inputs, labels in dataloader_test:
    y_pred = model(Variable(inputs))
    plt.imshow(labels[0])
    plt.show()
    print(y_pred[0])
    plt.imshow(y_pred[0].detach().numpy())
    plt.show()