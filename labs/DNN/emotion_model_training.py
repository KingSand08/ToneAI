import os
import numpy as np
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch.nn as nn
import torch

# Variables
train_slice = .75
validation_slice = .15
batch_size = 32


# Datasets
input_features_np = np.load(os.path.join(os.path.dirname(__file__), 'input_features.npy'), allow_pickle=True)
emotion_targets_np = np.loadtxt(os.path.join(os.path.dirname(__file__), 'emotion_targets.csv'), delimiter=',') 
input_features = torch.from_numpy(input_features_np)
emotion_targets = torch.from_numpy(emotion_targets_np)
dataset = TensorDataset(input_features, emotion_targets)


# Split model sets
train_size = int(train_slice * len(dataset))
val_size = int(validation_slice * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, emotion_targets.shape[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleDNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
        print(f'Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}')


    model.eval()
    with torch.no_grad():
        test_loss = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
        print(f'Test Loss: {test_loss/len(test_loader):.4f}')
