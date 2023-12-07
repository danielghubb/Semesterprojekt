import copy
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import h5py
from torch.utils.data import Dataset, DataLoader

##########Hyperparameter#############
batch_size = 32
lr = 0.01
epochen = 5
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
    nn.Linear(4096, 65536)
)

model.to(device)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr) #gradient decent, efficent with less memory use
criterion = nn.MSELoss()


# Hold the best model
best_mse = np.inf
eval_history = []
train_history = []


# Training Loop
"""
def train(model, train_dl, criterion, optimizer):
    model.train()
    for (inputs, targets) in train_dl:
        optimizer.zero_grad()
        inputs, targets = inputs.float().to(device), targets.float().to(device)
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
"""
# Evaluation loop
def eval_model(model, test_dl, criterion):
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_targets = []
        for inputs, targets in test_dl:
            output = model(inputs.to(device))
            all_predictions.append(output.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    return mean_squared_error(all_targets, all_predictions)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

def train_with_scheduler(model, train_dl, criterion, optimizer, scheduler):
    model.train()
    total_batches = len(train_dl)

    for epoch in range(epochen):
        epoch_losses = []
        with tqdm(total=total_batches, desc=f'Epoch {epoch + 1}/{epochen}', unit='batch') as pbar:
            for i, (inputs, targets) in enumerate(train_dl):
                optimizer.zero_grad()
                inputs, targets = inputs.float().to(device), targets.float().to(device)

                with torch.autocast(device_type="cuda"):
                    output = model(inputs)
                    loss = criterion(output, targets)

                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})

                epoch_losses.append(loss.item())

        # Calculate and print the average loss for the epoch
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1}: Average Loss = {avg_epoch_loss:.4f}")

# Logging and documentation
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logging()

train_dl, test_dl = dataloader_train, dataloader_test
logger.info(f"Training set size: {len(train_dl.dataset)}, Test set size: {len(test_dl.dataset)}")

logger.info("Training begins")
history = []
for epoch in range(1, epochen + 1):
    train_with_scheduler(model, train_dl, criterion, optimizer, scheduler)
    mse = eval_model(model, test_dl, criterion)
    history.append(mse)
    logger.info(f"Epoch {epoch}: Mean MSE = {mse}")

# Print predictions for a subset of the test set
for inputs, _ in test_dl:
    prediction = model(inputs.to(device))
    logger.info('Predicted: {}'.format(prediction.detach().cpu().numpy()))