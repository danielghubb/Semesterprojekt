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

        x_data_normal = x_data[:7]

        return x_data_normal, y_data

    def close(self):
        self.h5_file.close()

    def getmms(self):
        return self.scaler
    
class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()

        # Input layer for 7 variables
        self.input_layer = nn.Linear(7, 1024)
        self.act1 = nn.ReLU()
        # Fully connected layer to connect to the deconvolution layers
        self.fc_layer = nn.Linear(1024, 256 * 32 * 32)  # Adjusted to match the product of deconv1 input size
        self.act2 = nn.ReLU()
        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # Adjusted kernel_size and output channels
        self.act3 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # Adjusted output channels and kernel_size
        self.act4 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)    # Adjusted output channels and kernel_size
        self.act5 = nn.ReLU()

    def forward(self, x):
        # Input layer
        x = self.act1(self.input_layer(x))
        # Fully connected layer
        x = self.act2(self.fc_layer(x))
        x = x.reshape(-1, 256, 32, 32)  # Reshape to match deconv1 input size
        # Deconvolution layers
        x = self.act3(self.deconv1(x))
        x = self.act4(self.deconv2(x))
        x = self.act5(self.deconv3(x))
        x = x.view(-1, 256, 256).squeeze()

        return x

def train():
    ##########Hyperparameter#############
    file_path = r'./normed_data_2mio.h5'
    batch_size = 32
    lr = 0.001
    ##########Hyperparameter#############

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print(device)

    # initialise dataset and dataloader
    dataset = H5Dataset(file_path)

    train, test =  random_split(dataset, [0.66, 0.34])
    dataloader_train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=32)
    dataloader_test = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=32)

    # Accessing the first group in the dataset
    #sample_x, sample_y = dataset[0]
    #print(dataset[0][0])
    #print("X data:", sample_x)
    #print("Y data:", sample_y)

    for model_nr in [500]:
        model = DeconvNet().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

        n_epochs = model_nr
        history = []
        best_mse = np.inf 
        for epoch in range(n_epochs):
            print(f"epoch: {epoch}")
            i = 0
            mse = []
            for inputs, labels in dataloader_train:
                # forward, backward, and then weight update
                y_pred = model(inputs.to(device))
                loss = loss_fn(y_pred, labels.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mse.append(float(loss.cpu()))
                print(f"training batch: ({i}/{len(dataloader_train)})", end='\r', flush=True)
                i += 1
            i = 0
            model.to(device)
            print('\n')
            for inputs, labels in dataloader_test:
                y_pred = model(inputs.to(device))
                print(f"test batch: ({i}/{len(dataloader_test)})", end='\r', flush=True)
                i += 1
            if epoch in [1, 5, 20, 50, 100, 200, 300, 400, 500]:
                torch.save(model, './model' + '_checkpoint_' + str(epoch) + '.pth')
                plt.plot(history)
                plt.savefig('loss_' + str(epoch))
                plt.close()
            history.append(np.mean(mse))
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]
            print('\n')
            print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))

        model.cpu()

        torch.save(model, './model_' + str(model_nr) + '.pth')
        plt.plot(history)
        plt.savefig('loss_' + str(epoch))
        plt.close()
        print("MSE: %.5f" % best_mse)
        print("RMSE: %.5f" % np.sqrt(best_mse))

if __name__ == '__main__':
    train()
