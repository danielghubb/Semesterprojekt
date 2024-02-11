import copy
import time

import matplotlib.pyplot as plt
import numpy as np
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
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

for model_nr in [5, 10, 20, 50, 100, 200]:
    path = './data.csv'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data = pd.read_csv(path)

    # find constant columns, else normalization wouldn't work
    data = data.loc[:, (data != data.iloc[0]).any()] 

    # label input 0-6, output 0-4 just for clarity
    num_inputs = 7
    num_outputs = 5
    data.columns = np.concatenate(
        (["Input %s" % i for i in range(num_inputs)], ["Output %s" % i for i in range(num_outputs)])
    )

    scaler = MinMaxScaler()
    scaler.fit(data)
    normalized_data = scaler.transform(data)

    train, test = train_test_split(normalized_data, train_size=0.67, shuffle=True)

    x_train, y_train = train[:,:num_inputs], train[:,num_inputs:]
    x_test, y_test = test[:,:num_inputs], test[:,num_inputs:]

    print('data preparation done')

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 5)
    X_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 5)

    # Define the model
    model = nn.Sequential(
        nn.Linear(7, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 5)
    )

    # loss function and optimizer
    loss_fn = nn.MSELoss()  # mean square error
    lr = 0.01
    #lr = 0.005
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

    n_epochs = model_nr   # number of epochs to run
    batch_size = 32  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []

    for epoch in range(n_epochs):
        model.train().to(device)
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size].to(device)
                y_batch = y_train[start:start+batch_size].to(device)
                # forward pass
                #with torch.cuda.amp.autocast():
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward(retain_graph = True)
                # update weights
                optimizer.step()
                # print progress
                bar.update()
                bar.set_postfix(mse=float(loss))
            bar.close()
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]
            print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test).cpu()
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        print(f'MSE: {mse}')
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.cpu().state_dict())
        if epoch == int(n_epochs/2):
            torch.save(model, './model_' + str(model_nr) + '_checkpoint.pth')
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    torch.save(model, './model_' + str(model_nr) + '.pth')
    print("MSE: %.5f" % best_mse)
    print("RMSE: %.5f" % np.sqrt(best_mse))
    plt.plot(history)
    plt.savefig('loss_' + str(model_nr))

    #model = torch.load('./model2.pth')

    model.eval()
    model.cpu()

    nr_tests = 30

    #forward some inputs todo
    indices = torch.randperm(len(test))[:nr_tests]
    indices = indices.numpy()

    array_target = []
    array_pred =[]
    for i in range(0,nr_tests):
        data = X_test.cpu()[indices[i]]
        target = y_test.cpu()[indices[i]]
        array_target.append(target.numpy())

        y_pred = model(data)
        array_pred.append(y_pred.detach().numpy())


    null_matrix = np.zeros((nr_tests,7))
    matrix_target = np.reshape(array_target, (-1,5))
    matrix_pred = np.reshape(array_pred, (-1,5))

    matrix_target_full = np.append(null_matrix,matrix_target, axis =1)
    matrix_pred_full = np.append(null_matrix,matrix_pred, axis =1)

    matrix_target_revers = scaler.inverse_transform(matrix_target_full)[:,7:]
    matrix_pred_revers = scaler.inverse_transform(matrix_pred_full)[:,7:]

    # Plot the last 5 scatter plots as subplots in a single figure with 2 rows
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Define a color map for each feature
    colors = plt.cm.viridis(np.linspace(0, 1, 5))

    for i in range(5):
        row, col = divmod(i, 3)
        axs[row, col].scatter(matrix_target_revers[:, i], matrix_pred_revers[:, i],
                                label=f'Variable = {i + 1}', c=colors[i], alpha=0.8)
        axs[row, col].axline((0, 0), slope=1, color='k')
        axs[row, col].set_title(f'Output Variable {i + 1}')
        axs[row, col].set_xlabel('Target')
        axs[row, col].set_ylabel('Prediction')
        axs[row, col].legend()

    # Hide empty subplot
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('inference_' + str(model_nr))
