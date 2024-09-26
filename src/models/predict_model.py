import torch
import torch.nn as nn
    
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu, self.dropout)

    def forward(self, x):
        return self.net(x)

class TCNWeatherPredictor(nn.Module):
    def __init__(self, input_size=4, output_size=4, num_channels=64, kernel_size=3, output_window=30, dropout=0.5):
        super(TCNWeatherPredictor, self).__init__()
        self.temporal_block = TemporalBlock(input_size, num_channels, kernel_size, stride=1, dilation=2, padding=2, dropout=dropout)
        # self.fc1 = nn.Linear(num_channels * (90 - 2 * (kernel_size - 1)), 128)  # Adjust based on input length and kernel size
        self.fc1 = nn.Linear(11264, 128)  # Adjust based on input length and kernel size
        self.fc2 = nn.Linear(128, output_size * output_window)
        self.output_size = output_size
        self.output_window = output_window

    def forward(self, x):
        # TCN forward pass
        x = x.transpose(1, 2)  # Switch time dimension for conv1d (batch_size, input_size, time_steps)
        x = self.temporal_block(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, self.output_window, self.output_size)  # Reshape to output window
        return x