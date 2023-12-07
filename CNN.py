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

from randomSplit import random_split


##########Hyperparameter#############
#file_path = '/home/kali/Projects/Semesterprojekt/rzp-1_sphere1mm_train_100k.h5'
file_path = '/vol/fob-vol7/mi21/arendtda/Sempro/rzp-1_sphere1mm_train_100k.h5'
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
        self.fc_layer = nn.Linear(1024, 256 * 254 * 254)
        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose2d(256, 1, kernel_size=4, stride=2)

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        # Fully connected layer
        x = self.fc_layer(x)
        x = x.view(-1, 256, 254, 254)  # Reshape to (batch_size, 256, 254, 254)
        # Deconvolution layer
        x = self.deconv1(x)

        return x

# Create an instance of the DeconvNet
model = DeconvNet()

# Example input with batch size 1 and 7 input channels
input_data = Variable(dataset[0][0])
#print(input_data)
# Forward pass
output = model(input_data)

# Print the output shape
print("Output shape:", output.shape)
print(output)

# Closing the dataset
dataset.close()
