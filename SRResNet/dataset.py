import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CelebaHQDataSet(Dataset):
    def __init__(self, img_size_lr, img_size_hr, total_images=None):
        self.img_size_lr = img_size_lr
        self.img_size_hr = img_size_hr
        project_path = os.path.dirname(os.path.dirname(__file__)) # should be PSIML6/
        project_path = os.path.abspath(project_path)
        self.data_root = os.path.join(project_path, "SRResNet",  "data", "data256x256") #privremeno promenjeno!!!!!!
        self.image_paths = glob.glob(os.path.join(self.data_root, "*.jpg"))
        self.image_paths = self.image_paths[:total_images]
        self.transforms_lr = transforms.Compose(
            [
                transforms.Resize(self.img_size_lr),
                transforms.ToTensor() # this puts the image in the CxHxW format and normalizes it to [0,1). See if we want this!
            ]
        )
        self.transforms_hr = transforms.Compose(
            [
                transforms.ToTensor() # this puts the image in the CxHxW format and normalizes it to [0,1). See if we want this!
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path =  self.image_paths[index]
        image_lr = Image.open(image_path)
        image_hr = Image.open(image_path)
        return (self.transforms_lr(image_lr), self.transforms_hr(image_hr))

#print( os.path.dirname(os.path.dirname(__file__)))