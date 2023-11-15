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
import random
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler(feature_range=(0,1))



##########Hyperparameter###########
epochen = 30
lr = 0.00001
batchsize = 64
##########Hyperparameter###########

# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self):
        # store the inputs and outputs
        xy = mms.fit_transform(np.loadtxt("/home/sar-user/Projekte/Semesterprojekt/data.csv", delimiter = "," , dtype= np.float32))
        #normalisieren und zu tensor transformen
        #self.y = F.normalize(torch.from_numpy(xy[:,8:]), p=2.0, dim=1)
        #self.x = F.normalize(torch.from_numpy(xy[:,:8]), p=2.0, dim=1)
        self.y = torch.from_numpy(xy[:,8:])
        self.x = torch.from_numpy(xy[:,:8])
        
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
train_dl = DataLoader(train, batchsize, shuffle=True)
test_dl = DataLoader(test, batchsize, shuffle=False)


#Model Definition
model = nn.Sequential(
    nn.Linear(8, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 6)
)

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


null_matrix = np.zeros((9,8))
matrix_target = np.reshape(array_target, (-1,6))
matrix_pred = np.reshape(array_pred, (-1,6))

matrix_target_full = np.append(null_matrix,matrix_target, axis =1)
matrix_pred_full = np.append(null_matrix,matrix_pred, axis =1)

matrix_target_revers = mms.inverse_transform(matrix_target_full)[:,8:]
matrix_pred_revers = mms.inverse_transform(matrix_pred_full)[:,8:]


for i in range(0,6):
    plt.scatter(matrix_target[:,i], matrix_pred[:,i], label = f'Variable = {i + 1}')

plt.axline((0, 0), slope=1, color='k')
plt.title('Scatterplot nach Outputvariable')
plt.xlabel('Target')
plt.ylabel('Prediction')
plt.legend()
plt.show()


for i in range(0,6):
    plt.scatter(matrix_target_revers[:,i], matrix_pred_revers[:,i], label = f'Variable = {i + 1}')
    plt.axline((0, 0), slope=1, color='k')
    plt.title(f'Scatterplot nach Outputvariable {i+1}')
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.legend()
    plt.show()





