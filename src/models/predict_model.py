import torch
import torch.nn as nn

class CNNWeatherPredictor(nn.Module):
    def __init__(self, input_size=6, output_size=6, kernel_size=3, num_filters=64):
        super(CNNWeatherPredictor, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size)
        self.fc1 = nn.Linear(num_filters * (90 - 2 * (kernel_size - 1)), 128)  # Adjust depending on the kernel size and input length
        self.fc2 = nn.Linear(128, output_size * 30)  # Assuming 30-day prediction

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        # Input shape: (batch_size, time_steps, input_size)
        x = x.transpose(1, 2)  # Reshape to (batch_size, input_size, time_steps)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        # x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.relu(x)
        x = x.view(-1, 30, 6)  # Reshape to (batch_size, 30 days, 6 features)

        return x