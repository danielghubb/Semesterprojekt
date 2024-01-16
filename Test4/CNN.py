import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import h5py
import numpy as np

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

        x_data = torch.Tensor(group['X'][:7])
        y_data = torch.Tensor(group['Y'][:])

        return x_data, y_data

    def close(self):
        self.h5_file.close()

class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()

        self.input_layer = nn.Linear(7, 128)
        self.act1 = nn.ReLU()
        self.fc_layer = nn.Linear(128, 256)
        self.act2 = nn.ReLU()

        # Encoder with Convolutional Layers and Batch Normalization
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Decoder with Deconvolutional Layers and Batch Normalization
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(16)
        self.deconv4 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1)
        self.deconv7 = nn.ConvTranspose2d(2, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.act1(self.input_layer(x))
        x = self.act2(self.fc_layer(x))

        # Reshape to a 4D tensor for convolutional layers
        x = x.view(x.size(0), 1, 16, 16)

        # Encoder
        x1 = self.bn1(self.conv1(x))
        x2 = self.bn2(self.conv2(x1))
        x3 = self.bn3(self.conv3(x2))
        x4 = self.bn4(self.conv4(x3))

        # Decoder with skip connections
        x = self.bn5(self.deconv1(x4)) + x3
        x = self.bn6(self.deconv2(x)) + x2
        x = self.bn7(self.deconv3(x)) + x1
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        # print(x.shape)
        x = x.view(-1, 1, 256, 256)
        x = x.reshape(x.size(0), 256, 256)

        return x

def train():
    file_path = r'../Aufgabe2/data.h5'
    batch_size = 32
    lr = 0.001
    n_epochs = 200

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)

    dataset = H5Dataset(file_path)
    train, test = random_split(dataset, [0.66, 0.34])
    dataloader_train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=72)
    dataloader_test = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=72)

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
            loss = loss_fn(y_pred, labels.to(device))
            loss.backward()
            optimizer.step()
            mae.append(float(loss.cpu()))
            print(f"training batch: ({len(mae)}/{len(dataloader_train)})", end='\r', flush=True)

        print('\n')
        i = 1
        for inputs, labels in dataloader_test:
            y_pred = model(inputs.to(device))
            print(f"test batch: ({i}/{len(dataloader_test)})", end='\r', flush=True)
            i += 1

        if epoch % 10 == 0 or epoch == 1:
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
                
                # You might need to modify this based on your specific requirements
                # For example, if the output is an image, you can use imshow to display it.
                plt.imshow(sample_outputs[0].cpu().numpy())
                plt.title('Sample Inference')
                plt.savefig(f'./inference_plot_{epoch}.png')
                plt.close()

        history.append(np.mean(mae))
        scheduler.step()
        print('\n')
        print("Epoch %d: Learning Rate %.7f" % (epoch, optimizer.param_groups[0]["lr"]))

    torch.save(model.state_dict(), './model_final.pth')
    plt.plot(history)
    plt.title('Final Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_final.png')
    plt.close()

if __name__ == '__main__':
    train()
