import torch
import torchvision
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

#from critics import FCCritic
#from generators import FCGenerator
from dataset import CelebaHQDataSet

import torchvision.transforms as transforms

IMG_SIZE_LR = 64
IMG_SIZE_HR = 256
BATCH_SIZE = 10
NUM_WORKERS = 8

def train():
    data_set = CelebaHQDataSet(IMG_SIZE_LR, IMG_SIZE_HR)
    image = data_set.__getitem__(0)
    #print(image[0].shape)
    #print(image[0])
    #print(image[1].shape)
    #print(image[1])

    data_loader = DataLoader(data_set,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=NUM_WORKERS,
                            drop_last=True)

    #print(data_loader[0].shape)
    batch = next(iter(data_loader))
    #print(batch[0])
    print(batch[0][0])
    print(batch[1][0])
    print(batch[0][0].shape)
    print(batch[1][0].shape)
    print(len(batch))

if __name__ == "__main__":
    train()