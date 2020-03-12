import layers
import torch
import torch.nn as nn 
import torch.nn.functional as F

# Vanilla definition
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.batchNorm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batchNorm2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.batchNorm1(self.conv1(x))))
        x = self.pool(F.relu(self.batchNorm2(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Network definition with blocks
class NetworkWithBlock(nn.Module):
    def __init__(self):
        super(NetworkWithBlock, self).__init__()
        self.block1 = layers.ConvBatchMaxPool(1,6,5,2)
        self.block2 = layers.ConvBatchMaxPool(6,16,5,2)
        self.block3 = layers.LinearRelu(16*4*4,256)
        self.block4 = layers.LinearRelu(256,84)
        self.fc = nn.Linear(84, 10)

    def forward(self, x):
        
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.block3(x)
        x = self.block4(x)
        x = self.fc(x)

        return x

# Random
class NetworkWithXBlock(nn.Module):
    def __init__(self,in_channels,extractor_depth):
        super(NetworkWithXBlock, self).__init__()
        self.extractor = []
        for i in range(extractor_depth):
            self.extractor.append(layers.ConvBatchMaxPool(in_channels,in_channels*2,2,2))
            in_channels = in_channels*2
        self.layers = nn.Sequential(*self.extractor)

    def forward(self, x):
        x = self.layers(x)
        return x