import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from torch.autograd import Variable

from Modeli import *
from dataset import CelebaHQDataSet

import torch.nn as nn

import torch

IMG_SIZE_LR = 64    #velicine slika
IMG_SIZE_HR = 256
BATCH_SIZE = 16
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ada1 = 0.9          #koeficijenti za adam optimizer
ada2 = 0.999

LR = 0.0002 #learning rate


EPOCH = 100000

def train():
    cuda = torch.cuda.is_available()

    hr_shape = (IMG_SIZE_HR, IMG_SIZE_HR)


    generator = ResNetG()
    vgg = VGG()


    vgg.eval()

    MeanAbsErr = torch.nn.L1Loss()

    if cuda:
        generator = generator.cuda()
        vgg = vgg.cuda()
        MeanAbsErr = MeanAbsErr.cuda()





    optimizer = torch.optim.Adam(generator.parameters(), LR, betas=(ada1, ada2))


    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    dataset = CelebaHQDataSet(IMG_SIZE_LR, IMG_SIZE_HR)
    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=NUM_WORKERS,
                            drop_last=True)



    #for i, real_img_batch in tqdm(enumerate(data_loader), total=total_iterations, desc=f"Epoch: {epoch}", unit="batches"):
    #imgs -> img_batch
    # ----------
    #  Training
    # ----------

    total_iterations = len(dataset) // BATCH_SIZE

    for epoch in range(EPOCH):
        for i, img_batch in tqdm(enumerate(dataloader), total=total_iterations, desc=f"Epoch: {epoch}", unit="batches"):

            imgs_lr = img_batch[0].to(DEVICE)
            imgs_hr = img_batch[1].to(DEVICE)   #UCITAJ HIGH I LOW RESOLUTION IMAGES< TREBA DA BUDU NORMALIZOVANE NA INTERVALU 0 1


            # ------------------
            #  Train Generators
            # ------------------

            optimizer.zero_grad()

            gen_hr = generator(imgs_lr)


            #loss funkcija sadrzaja
            gen_features = vgg(gen_hr)
            real_features = vgg(imgs_hr)
            loss_content = MeanAbsErr(gen_features, real_features.detach())

            loss = loss_content

            loss.backward()
            optimizer.step()

            #OVDE Treba neki logging staviti ima tqdm ili mozemo onaj tensor info ili tako nesto sto loguje na loalhost

if __name__ == "__main__":
    train()
