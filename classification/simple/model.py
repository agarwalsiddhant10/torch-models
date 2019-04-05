import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as numpy

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, 1, 2)
        self.conv2 = nn.Conv2d(8, 32, 5, 1, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.2)
        


    def forward(self, X):
        # print(X.shape)
        X = F.max_pool2d(F.relu(self.conv1(X)), 2) 
        X = self.dropout1(X)
        # print(X.shape)
        X = F.max_pool2d(F.relu(self.conv2(X)), 2) 
        X = self.dropout2(X)
        # print(X.shape)
        X = F.max_pool2d(F.relu(self.conv3(X)), 2) 
        X = self.dropout3(X)
        # print(X.shape)
        X = X.view(X.shape[0],-1)
        # print(X.shape)

        X = F.relu(self.fc1(X))
        X = self.dropout4(X)

        X = self.fc2(X)

        return X

# net = Net()
# a = torch.ones((8, 3, 32, 32))
# net.eval()
# Y = net.forward(a)
