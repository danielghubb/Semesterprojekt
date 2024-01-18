import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random

from clustering2 import cluster
import optuna



#Hyperparameter
model_path = '/vol/fob-vol7/mi21/arendtda/Sempro/model_checkpoint_150.pth'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
ySize = 256

kernel_size = 6
stride = 4
padding = 1

class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(7, 256),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=kernel_size, stride=stride, padding=padding),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 1, 1)
        x = self.decoder(x)
        return x


model = DeconvNet()
model.load_state_dict(torch.load(model_path))
model.to(device)
#model.load_state_dict(torch.load(model_path, map_location= torch.device('cpu')))
model.eval()

def objective(trial):
    # Suggest values for each element of the tensor within a reasonable range
    tensor_values = [[trial.suggest_uniform(f'tensor_value_{i}', 0.0, 1.0) for i in range(7)]] #this is not stupid because it works
    tensor_values= torch.Tensor(tensor_values, device= torch.device("cpu"))

    inputs = tensor_values.to(device)
    y_pred = model(inputs).cpu().detach().numpy().reshape(-1, ySize, ySize)

    # Methode 1: clustering auf 2D Vektor mit k-means
    objective_value = cluster(y_pred ,1 ,ySize ,2 )

    return 1 - objective_value

study = optuna.create_study(direction='minimize')  # minimize the objective value
study.optimize(objective, n_trials=1000)

best_tensor_values = [[study.best_params[f'tensor_value_{i}'] for i in range(7)]]
print(best_tensor_values)
best_tensor_values= torch.Tensor(best_tensor_values, device= torch.device("cpu"))
inputs = best_tensor_values.to(device)

y_pred = model(inputs).cpu().detach().numpy().reshape(-1, ySize, ySize)

fig, axes = plt.subplots(1, 1, figsize=(12, 5))

plt.imshow(y_pred[0])
plt.title('Best Model Output')

# Set the same size for both subplots
#axes[0].set_aspect('equal', adjustable='datalim')
#axes[1].set_aspect('equal', adjustable='datalim')

# Adjust layout
#plt.tight_layout()
# Show the plots
plt.show()
