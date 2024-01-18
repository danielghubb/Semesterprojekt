import copy
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import pandas as pd
from scipy import datasets
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim.lr_scheduler as lr_scheduler
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable

##########Hyperparameter#############
batch_size = 32
lr = 0.01
epochen = 10
##########Hyperparameter#############

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

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
        y_data = torch.Tensor(group['Y'][:]).flatten()

        x_data_normal = normalize(x_data[:7])

        return x_data_normal, y_data

    def close(self):
        self.h5_file.close()

    def getmms(self):
        return self.scaler

# initialise dataset and dataloader
file_path = r'C:\Users\fagda\Documents\Semesterprojekt\Aufgabe2\data.h5'

dataset = H5Dataset(file_path)

train, test =  torch.utils.data.random_split(dataset, [0.66, 0.34])
dataloader_train = DataLoader(train, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(test, batch_size=batch_size, shuffle=False)

#Model Definition
model = nn.Sequential(
    nn.Linear(7, 512),
    nn.ReLU(),
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 4096),
    nn.ReLU(),
    nn.Linear(4096, 8192),
    nn.ReLU(),
    nn.Linear(8192, 16384),
    nn.ReLU(),
    nn.Linear(16384, 32768),
    nn.ReLU(),
    nn.Linear(32768, 65536)
)

model.to(device)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr) #gradient decent, efficent with less memory use


# Hold the best model
best_mse = np.inf
eval_history = []
train_history = []


#Trainieren
def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate (dataloader_train):
        data = Variable(data).to(device)
        target = Variable(target).to(device)

        #forward pass
        out = model(data)
        loss = loss_fn(out, target)
        
        #backward pass
        optimizer.zero_grad()
        loss.backward()

        #update
        optimizer.step()



def evaluate(epoch):
    model.eval()
    epoch_mse = []
    for batch_id, (data, target) in enumerate (dataloader_test):
        data = Variable(data).to(device)
        target = Variable(target).to(device)

        y_pred = model(data)
        mse = loss_fn(y_pred, target)
        mse = float(mse)
        epoch_mse.append(mse)
    
    eval_history.append(np.mean(epoch_mse))



def evaluate_train_data(epoch):
    model.eval()
    epoch_mse_train = []
    for batch_id, (data, target) in enumerate (dataloader_train):
        data = Variable(data).to(device)
        target = Variable(target).to(device)

        y_pred = model(data)
        mse = loss_fn(y_pred, target)
        mse = float(mse)
        epoch_mse_train.append(mse)
    
    print(np.mean(epoch_mse_train))
    train_history.append(np.mean(epoch_mse_train))
    




print("Training beginnt")

for epoch in range (1, epochen+1):
    print("Epoch: " + str(epoch))

    #training, Evaluation
    train(epoch)
    evaluate_train_data(epoch)
    evaluate(epoch)

    #bestes Modell speichern
    if epoch == 1:
        torch.save(model.state_dict(), 'model_checkpoint.pth')
    else:
        if eval_history[-1] < eval_history[-2]:
            torch.save(model.state_dict(), 'model_checkpoint.pth')
 

# plot MSE Loss on Train and Test Data
plt.plot(eval_history, color='red', label='MSE loss test data')
plt.plot(train_history, color='blue', label='MSE loss train data')
plt.xlabel('Epoche')
plt.ylabel('Mean MSE Loss')
plt.legend()
plt.show()
plt.savefig('foo.png')

# Closing the dataset
dataset.close()
