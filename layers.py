import torch
import torch.nn as nn 
import torch.nn.functional as F

# Linear Block
class LinearRelu(nn.Module):
    def __init__(self,in_features, out_features):
        super(LinearRelu,self).__init__()
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self,x):
        x = self.fc(x)
        x = F.relu(x)
        return x 

# Conv Batch MaxPool Block
class ConvBatchMaxPool(nn.Module):
    
    def __init__(self,in_channels, nb_filtre, kernel_size,pool_size):
        super(ConvBatchMaxPool,self).__init__()
        self.conv = nn.Conv2d(in_channels, nb_filtre, kernel_size)
        self.batchnorm = nn.BatchNorm2d(nb_filtre)
        self.pool = nn.MaxPool2d(pool_size)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.pool(x)
        return x 