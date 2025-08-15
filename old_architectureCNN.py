from imports import *

class OldConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        if isinstance(kernel_size, tuple):
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            padding = kernel_size // 2

        super(OldConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class OldChordCNN(nn.Module):
    def __init__(self, channels, num_classes, dropout=0.5):
        super(OldChordCNN, self).__init__()
        self.conv1 = OldConvBlock(channels[0], channels[1], (3, 3))
        self.conv2 = OldConvBlock(channels[1], channels[1], (3, 3))
        self.conv3 = OldConvBlock(channels[1], channels[1], (3, 3))
        self.pool1 = nn.MaxPool2d((1, 2))

        self.conv4 = OldConvBlock(channels[1], channels[3], (9, 12))
        self.dropout = nn.Dropout(dropout)
        self.conv5 = OldConvBlock(channels[3], channels[1], (3, 3))
        self.conv6 = OldConvBlock(channels[1], channels[2], (3, 3))
        self.pool2 = nn.MaxPool2d((1, 2))

        self.conv7 = OldConvBlock(channels[2], channels[2], (3, 3))
        self.conv8 = nn.Conv2d(channels[2], 25, (1, 1))  # Última camada convolucional

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(25, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = self.dropout(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool2(x)

        x = self.conv7(x)
        x = self.conv8(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
