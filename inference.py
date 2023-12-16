import copy
from os import write
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import MinMaxScaler

from randomSplit import random_split





##########Hyperparameter#############
file_path = r'normed_data.h5'
#file_path = r'/vol/fob-vol7/mi21/arendtda/Sempro/normed_data.h5'
model_path = r'C:\Users\fagda\Downloads\model_checkpoint_5.pth'
#model_path = '/vol/fob-vol7/mi21/arendtda/Sempro/model_checkpoint_5.pth'
batch_size = 32
##########Hyperparameter#############

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)


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

        return x_data, y_data

    def close(self):
        self.h5_file.close()


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

minsY = [0, 17211]
data = np.zeros((256, 256), float)
data[0][0] = 0
data[0][1] = 17211

# Flatten the 2D array before fitting to MinMaxScaler
data_flat = data.flatten().reshape(-1, 1)

scaler = MinMaxScaler()
scaler.fit(data_flat)

model = torch.load(model_path).to(device)

for inputs, labels in dataloader_test:
    y_pred = model(inputs.to(device)).cpu()
    y_pred = scaler.inverse_transform(y_pred[0].detach().numpy().flatten().reshape(-1, 1)).reshape(256, 256)

    plt.imshow(labels[0])
    plt.show()
    print(y_pred)
    plt.imshow(y_pred)
    plt.show()
