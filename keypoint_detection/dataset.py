import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2

class dataset(Dataset):
    def __init__(self, image_folder, annotation_file):
        self.images = image_folder
        self.annotations = annotation_file

        self.coco = COCO(self.annotations)
        self.catId = self.coco.getCatIds(catNms = ['person'])

        self.imgIds = self.coco.getImgIds(catIds = self.catId)
        
    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, id):
        # print('yahan baar baar nahi aana chahiye')
        print(self.imgIds[id])
        img = self.coco.loadImgs(self.imgIds[id])[0]
        img_name = img['file_name']
        # print(img_name)
        X = np.array(cv2.imread(self.images + str(img_name)))
        height = X.shape[0]
        width = X.shape[1]
        # print(height, width)
        X = cv2.resize(X,(224, 224))
        # print(X.shape)
        X = X.transpose([2,0,1])
        annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=self.catId, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        y = anns[0]['keypoints']
        y = np.array(y)
        # print(y)
        for i in range(17):
            y[3*i] = 224*1.0/width * y[3*i]
            y[3*i + 1] = 224*1.0/height * y[3*i + 1]
        # print(y)
        return X, y
data_train = dataset('coco/images/', 'coco/annotations/person_keypoints_train2014.json')
train_loader =  DataLoader(data_train, batch_size = 8, shuffle = True)
print(len(train_loader))
data_train.__getitem__(5)


        