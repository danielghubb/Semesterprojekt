import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import h5py
import numpy as np

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

def train():
    file_path = r'normed_data_2mio_bin32.h5'
    batch_size = 32 #batchsize 32
    lr = 0.001 #0.001
    n_epochs = 200


    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)

    dataset = H5Dataset(file_path)
    
    dataset_size = len(dataset)
    train_size = int(0.66 * dataset_size)
    test_size = dataset_size - train_size
    train, test = random_split(dataset, [train_size, test_size])

    dataloader_train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=72, pin_memory = True)
    dataloader_test = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=72, pin_memory = True)

    model = DeconvNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    history = []

    for epoch in range(n_epochs):
        print(f"epoch: {epoch}")
        mse = []
        for batch_idx, (inputs, labels) in enumerate(dataloader_train):
            y_pred = model(inputs.to(device))
            loss = loss_fn(y_pred, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mse.append(float(loss.cpu()))
            print(f"training batch: ({batch_idx}/{len(dataloader_train)})", end='\r', flush=True)

        print('\n')
        for batch_idx, (inputs, labels) in enumerate(dataloader_test):
            y_pred = model(inputs.to(device))
            print(f"test batch: ({batch_idx}/{len(dataloader_test)})", end='\r', flush=True)

        if epoch % 25 == 0:
            torch.save(model.state_dict(), f'./model_checkpoint_{epoch}.pth')

            # Plot and save the loss curve
            plt.plot(history)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(f'./loss_plot_{epoch}.png')
            plt.close()

             # Plot and save some sample inferences
            with torch.no_grad():
                sample_inputs, _ = next(iter(dataloader_test))
                sample_inputs = sample_inputs.to(device)
                sample_outputs = model(sample_inputs)
                
                plt.imshow(sample_outputs[0].cpu().numpy())
                plt.title('Inference')
                plt.savefig(f'./inference_plot_{epoch}.png')
                plt.close()

        if epoch == 1000:
            break  # Stop training after 1000 epochs

        history.append(np.mean(mse))
        scheduler.step()
        print('\n')
        print("Epoch %d: Learning Rate %.4f" % (epoch, optimizer.param_groups[0]["lr"]))

    torch.save(model.state_dict(), './model_final.pth')
    plt.plot(history)
    plt.title('Final Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_final.png')
    plt.close()

if __name__ == '__main__':
    train()
