import os
from matplotlib import transforms
import numpy as np
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch.nn as nn
import torch


# Variables
train_slice = .70
test_slice = .15
validation_slice = .15
batch_size = 32


# Datasets
script_dir = os.path.join(os.path.dirname(__file__))
input_features_path = os.path.join(script_dir, 'input_features.pt')
input_features = torch.load(input_features_path).float()
emotion_targets_np = np.loadtxt(os.path.join(script_dir, 'emotion_targets.csv'), delimiter=',') 
emotion_targets = torch.from_numpy(emotion_targets_np)
dataset = TensorDataset(input_features, emotion_targets)

# Split model sets (15:validation 70:training, 15: testing)
total_len = len(dataset)
train_len = int(total_len * train_slice)
val_len = int(total_len * validation_slice)
test_len = total_len - train_len - val_len

# Split model sets (70:training, 15:validation, 15: testing)
train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, 
    [train_len, val_len, test_len]
)

# Load datasets to train model
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Create design DNN model for Tone AI
class ToneAIModelDNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        # define different layers
        self.lin = nn.
    
    def forward(self, x):
        return self.(x)
    