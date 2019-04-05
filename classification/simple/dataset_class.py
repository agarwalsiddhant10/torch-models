import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self, image_file, label_file, transform = None):
        self.labels = np.load(label_file)
        self.images = np.load(image_file)
        # print(self.images.shape)
        self.images = self.images.transpose([0,3,1,2])
        # print(self.images.shape)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        x = self.images[index]
        if(self.transform != None):
            x = self.transform(x)
        y = self.labels[index]

        return x, y

# data = dataset("labels.npy", "images.npy")
# trainloader = DataLoader(data, batch_size = 8, shuffle = True)
# print(len(trainloader))
