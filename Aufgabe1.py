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
        self.y = F.normalize(torch.from_numpy(xy[:,8:]), p=2.0, dim=1)
        self.x = F.normalize(torch.from_numpy(xy[:,:8]), p=2.0, dim=1)
        self.n_samples = xy.shape[0]
        

    # number of rows in the dataset
    def __len__(self):
        return self.n_samples

    # get a row at an index
    def __getitem__(self, idex):
        return self.x[idex], self.y[idex]
    

dataset = CSVDataset()
#print (dataset[1,:])
#Test ob einlesen funktioniert
#f_data = dataset[0]
#feat, label = f_data
#print(feat, label)

# select rows from the dataset
train, test =  torch.utils.data.random_split(dataset, [0.66, 0.34])
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
best_mse = np.inf
eval_history = []
train_history = []


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
        optimizer.zero_grad()
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
    eval_history.append(np.mean(epoch_mse))



def evaluate_train_data(epoch):
    model.eval()
    epoch_mse_train = []
    for batch_id, (data, target) in enumerate (train_dl):
        data = Variable(data)
        target = Variable(target)

        y_pred = model(data)
        mse = loss_fn(y_pred, target)
        mse = float(mse)
        epoch_mse_train.append(mse)
    
    print(np.mean(epoch_mse_train))
    train_history.append(np.mean(epoch_mse_train))
    




print("Training beginnt")

for epoch in range (1, 15):
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


#load best Model todo
#kwargs = {}
#args =[]
#best_model = model(*args, **kwargs)
#best_model.load_state_dict(torch.load('model_checkpoint.pth'))

#forward some inputs todo
indices = torch.randperm(len(test))[:10]
indices = indices.numpy()

array_target = []
array_pred =[]
for i in range(0,9):
    data = Variable(test[indices[i]][0])
    target = Variable(test[indices[i]][1])
    array_target.append(target.numpy())

    y_pred = model(data)
    array_pred.append(y_pred.detach().numpy())

matrix_target = np.reshape(array_target, (-1,6))
matrix_pred = np.reshape(array_pred, (-1,6))

print(matrix_target)
print(matrix_pred)

for i in range(0,6):
    plt.scatter(matrix_target[:,i], matrix_pred[:,i], label = f'Variable = {i + 1}')

plt.title('Scatterplot nach Outputvariable')
plt.xlabel('Target')
plt.ylabel('Prediction')
plt.legend()
plt.show()
