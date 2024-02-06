import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import h5py
import numpy as np

from randomSplit import random_split

CPUdevice= torch.device("cpu")
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
        
        x_data = torch.Tensor(group['X'][:], device = CPUdevice) ####
        y_data = torch.where(torch.isinf(torch.Tensor(group['Y'][:], device = CPUdevice)), torch.tensor(0.0), torch.Tensor(group['Y'][:], device = CPUdevice))

        return x_data[:7], y_data

    def close(self):
        self.h5_file.close()


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



def train():
    file_path = '/vol/fob-vol7/mi21/arendtda/Sempro/log_normed_data_2mio.h5'
    #file_path = '/home/kali/Projects/Semesterprojekt/rzp-1_sphere1mm_train_100k.h5'
    batch_size = 32
    lr = 0.001
    n_epochs = 1000

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    dataset = H5Dataset(file_path)
    train, test = random_split(dataset, [0.66, 0.34])
    dataloader_train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=72)
    #dataloader_test = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)

    model = DeconvNet().to(device)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    history = []

    for epoch in range(n_epochs):
        print(f"epoch: {epoch}")
        mae = []
        for inputs, labels in dataloader_train:
            optimizer.zero_grad()
            y_pred = model(inputs.to(device))
            loss = loss_fn(y_pred, labels.to(device).unsqueeze(1))
            loss.backward()
            optimizer.step()
            mae.append(float(loss.cpu()))
            print(f"training batch: ({len(mae)}/{len(dataloader_train)})", end='\r', flush=True)


        if epoch % 50 == 0 or epoch == 1:
            torch.save(model.state_dict(), f'./model_checkpoint_{epoch}.pth')

            # Plot and save the loss curve
            plt.plot(history)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(f'./loss_plot_{epoch}.png')
            plt.close()

        history.append(np.mean(mae))
        scheduler.step()
        print('\n')
        print("Epoch %d: Learning Rate %.6f" % (epoch, optimizer.param_groups[0]["lr"]))


if __name__ == '__main__':
    train()
