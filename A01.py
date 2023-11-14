from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)

# Hyperparameters
path = r'XFEL_KW0_Results_2.csv'
batch_size = 32
learn_rate = 0.001
train_ratio = 0.66
epochs = 2

# Dataset definition
class CustomDataset(Dataset):
    def __init__(self, path):
        data = np.genfromtxt(path, delimiter=",", dtype=np.float32)
        self.x = torch.tensor(data[:, :8])
        self.y = torch.tensor(data[:, 8:])
        #self.x = nn.BatchNorm1d(8)(torch.tensor(data[:, :8]))
        #self.y = nn.BatchNorm1d(6)(torch.tensor(data[:, 8:]))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return [self.x[index], self.y[index]]

# Model definition
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.batch_norm_input = nn.BatchNorm1d(8).to(device)
        self.batch_norm_output = nn.BatchNorm1d(6).to(device)
        self.model = nn.Sequential(
            nn.Linear(8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)
        ).to(device)

    def forward(self, x):
        x = self.batch_norm_input(x)
        x = self.model(x)
        return self.batch_norm_output(x)


# Data preparation
def prep_data(path, batch_size, train_ratio):
    csv_dataset = CustomDataset(path)
    train_size = int(train_ratio * len(csv_dataset))
    test_size = len(csv_dataset) - train_size
    train, test = torch.utils.data.random_split(csv_dataset, [train_size, test_size])
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_dl, test_dl

# Model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learn_rate)

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

    for epoch in range(epochs):
        epoch_losses = []
        with tqdm(total=total_batches, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for i, (inputs, targets) in enumerate(train_dl):
                optimizer.zero_grad()
                inputs, targets = inputs.float().to(device), targets.float().to(device)

                with torch.cuda.amp.autocast():
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

train_dl, test_dl = prep_data(path, batch_size, train_ratio)
logger.info(f"Training set size: {len(train_dl.dataset)}, Test set size: {len(test_dl.dataset)}")

logger.info("Training begins")
history = []
for epoch in range(1, epochs + 1):
    train_with_scheduler(model, train_dl, criterion, optimizer, scheduler)
    mse = eval_model(model, test_dl, criterion)
    history.append(mse)
    logger.info(f"Epoch {epoch}: Mean MSE = {mse}")

# Print predictions for a subset of the test set
for inputs, _ in test_dl:
    prediction = model(inputs.to(device))
    logger.info('Predicted: {}'.format(prediction.detach().cpu().numpy()))
