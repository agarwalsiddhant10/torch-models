import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F

num_classes = 51
class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, X):
        X = self.fc1(X)

        return X

# model = my_model()

