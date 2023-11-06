import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import copy
import matplotlib.pyplot as plt


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self):
        # store the inputs and outputs
        xy = np.loadtxt("/home/sar-user/Projekte/Semesterprojekt/data.csv", delimiter = "," , dtype= np.float32)
        #normalisieren und zu tensor transformen
        #self.y = torch.from_numpy(xy[:,8:])
        #self.x = torch.from_numpy(xy[:,:8])
        self.y = F.normalize(torch.from_numpy(xy[:,8:]), p=2.0, dim=0)
        self.x = F.normalize(torch.from_numpy(xy[:,:8]), p=2.0, dim=0)
        self.n_samples = xy.shape[0]
        

    # number of rows in the dataset
    def __len__(self):
        return self.n_samples

    # get a row at an index
    def __getitem__(self, idex):
        return self.x[idex], self.y[idex]
    

dataset = CSVDataset()
#Test ob einlesen funktioniert
#f_data = dataset[0]
#feat, label = f_data
#print(feat, label)

# select rows from the dataset
train, test =  torch.utils.data.random_split(dataset, [0.8, 0.2])
#print(len(train))
#print(len(test))

# create a data loader for train and test sets
train_dl = DataLoader(train, batch_size=64, shuffle=True)
test_dl = DataLoader(test, batch_size=64, shuffle=False)


#Model Definition
model = nn.Sequential(
    nn.Linear(8, 32),
    nn.ReLU(),
    nn.Linear(32, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6)
)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.000001) #gradient decent, efficent with less memory use


# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []


#Trainieren
def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate (train_dl):
        data = Variable(data)
        target = Variable(target)

        #forward pass
        out = model(data)
        loss = loss_fn(out, target)
        
        #backward pass
        optimizer.zero_grad
        loss.backward()

        #update
        optimizer.step()


def evaluate(epoch):
    model.eval()
    epoch_mse = []
    for batch_id, (data, target) in enumerate (test_dl):
        data = Variable(data)
        target = Variable(target)

        y_pred = model(data)
        mse = loss_fn(y_pred, target)
        mse = float(mse)
        epoch_mse.append(mse)
    
    print(np.mean(epoch_mse))
    history.append(np.mean(epoch_mse))

        #if mse < best_mse:
        #    best_mse = mse
        #    best_weights = copy.deepcopy(model.state_dict())
    




print("Training beginnt")
epoch = 1

for epoch in range (1, 30):
    print("Epoch: " + str(epoch))
    train(epoch)
    evaluate(epoch)
 

# restore model and return best accuracy
#model.load_state_dict(best_weights)
#print("MSE: %.2f" % best_mse)
#print("RMSE: %.2f" % np.sqrt(best_mse))
print(history)
type(history)
plt.plot(history)
plt.show