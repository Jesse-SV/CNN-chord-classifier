from imports import *
from CNNBlock import CNNBlock

# model = ChordCNN([1, 32, 64, 128], num_classes=len(CLASSES), dropout=dropout).to(device)

class ChordCNN(nn.Module):
    def __init__(self, dims, num_classes, dropout):
        super().__init__()
        self.layers = nn.ModuleList([CNNBlock(dims[i - 1], dims[i]) for i in range(1, len(dims))])
        self.dropout = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

'''
Camadas
        self.layer1 = CNNBlock(1, 32)
        self.layer2 = CNNBlock(32, 64)
        self.layer3 = CNNBlock(64, 128)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)              # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)    # [B, 128]
        x = self.dropout(x)
        x = self.fc(x)               # [B, num_classes]
        return x
'''