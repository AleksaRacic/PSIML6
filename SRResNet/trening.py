import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.autograd import Variable

from torch.data import DataLoader
from Modeli import *


import torch.nn as nn

import torch

HR_SIZE = 256
SR_SIZE = 64

ada1 = 0.9          #koeficijenti za adam optimizer
ada2 = 0.999

LR = 0.0002 #learning rate


EPOCH = 100000


cuda = torch.cuda.is_available()

hr_shape = (HR_SIZE, HR_SIZE)


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

# dataloader = DataLoader(
#     ovde treb podaci da se ucitaju
# )




# ----------
#  Training
# ----------

for epoch in range(EPOCH):
    for i, imgs in enumerate(dataloader):

        # imgs_lr = 
        # imgs_hr =    #UCITAJ HIGH I LOW RESOLUTION IMAGES< TREBA DA BUDU NORMALIZOVANE NA INTERVALU 0 1


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