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
file_path = r'binned_data.h5'
#file_path = r'/vol/fob-vol7/mi21/arendtda/Sempro/normed_data.h5'
model_path = r'model_checkpoint_375.pth'
#model_path = '/vol/fob-vol7/mi21/arendtda/Sempro/model_checkpoint_5.pth'
batch_size = 32
##########Hyperparameter#############

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)

# Set a consistent random seed for reproducibility
torch.manual_seed(42)

kernel_size1 = 6
kernel_size2 = 5
stride1 = 4
stride2 = 3
padding1 = 2

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

        x_data = torch.Tensor(group['X'][:7])
        y_data = torch.Tensor(group['Y'][:])

        # x_data = (x_data - x_data.min()) / (x_data.max() - x_data.min())

        return x_data, y_data

    def close(self):
        self.h5_file.close()

class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()

        self.fully_connected = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
           nn.ConvTranspose2d(256, 128, kernel_size=kernel_size1, stride=stride1, padding=padding1),
           nn.BatchNorm2d(128),
           nn.ReLU(),
           nn.ConvTranspose2d(128, 64, kernel_size=kernel_size1, stride=stride1, padding=padding1),
           nn.BatchNorm2d(64),
           nn.ReLU(),
           nn.ConvTranspose2d(64, 32, kernel_size=kernel_size1, stride=stride1, padding=padding1),
           nn.BatchNorm2d(32),
           nn.ReLU(),
           nn.ConvTranspose2d(32, 1, kernel_size=kernel_size2, stride=stride2, padding=padding1),   
           nn.BatchNorm2d(1),
           nn.ReLU(),
        )

    def forward(self, x):
        x = self.fully_connected(x)
        x = x.view(x.size(0), 256, 1, 1)
        x = self.deconv(x)
        x = x.view(-1, 1, 64, 64)
        x = x.reshape(x.size(0), 64, 64)

        return x

# initialise dataset and dataloader
dataset = H5Dataset(file_path)

train, test = random_split(dataset, [0.66, 0.34])
dataloader_train = DataLoader(train, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(test, batch_size=batch_size, shuffle=False)

model = DeconvNet()
model.load_state_dict(torch.load(model_path))
model.to(device)

# Function to calculate accuracy
def calculate_accuracy(predictions, targets):
    return np.sum(predictions == targets) / np.prod(targets.shape)

# Plotting 2 labels and predictions in one file with plots closer together
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

for i, (inputs, labels) in enumerate(dataloader_test):
    if i == 2:
        break

    inputs, labels = inputs.to(device), labels.numpy()

    # Model prediction
    y_pred = model(inputs).cpu().detach().numpy().reshape(-1, 64, 64)

    # Calculate accuracy
    accuracy = calculate_accuracy(y_pred, labels)

    # Plotting
    axes[i, 0].imshow(labels[0])
    axes[i, 0].set_title(f'Sample {i + 1}\nGround Truth')

    axes[i, 1].imshow(y_pred[0])
    axes[i, 1].set_title(f'Model Prediction\nAccuracy: {accuracy:.2%}')

plt.tight_layout(h_pad=2)
plt.show()
