from imports import *

class CNNBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.relu(x)
        return x