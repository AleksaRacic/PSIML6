import argparse
import os
import numpy as np
import math
import random
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
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
MAX_SUMMARY_IMAGES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ada1 = 0.9          #koeficijenti za adam optimizer
ada2 = 0.999

LR = 0.0002 #learning rate

def test():
    summary_writer = SummaryWriter()
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


    generator.state_dict = torch.load("resnet_model0.pt")
    generator.eval()

    dataset = CelebaHQDataSet(IMG_SIZE_LR, IMG_SIZE_HR)
    train_set, valid_set, test_set = random_split(dataset, [28000, 1000, 1000])
    test_dataloader = DataLoader(test_set,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=NUM_WORKERS,
                                drop_last=True)

    print("testiram")

    with torch.no_grad():
        test_loss_sum = 0     
        test_num = 0
        test_images = []
        rand_num = random.randint(0,10)
        for i, test_img_batch in enumerate(test_dataloader):
            test_num = i
            test_imgs_lr = test_img_batch[0].to(DEVICE)
            test_imgs_hr = test_img_batch[1].to(DEVICE) 
            test_gen_hr = generator(test_imgs_lr)
            test_gen_features = vgg(test_gen_hr)
            test_real_features = vgg(test_imgs_hr)
            test_loss = MeanAbsErr(test_gen_features, test_real_features.detach())
            test_loss_sum += test_loss
            if i == rand_num:
                test_images = test_gen_hr

        test_loss_mean = test_loss_sum / ((test_num + 1))
        summary_writer.add_scalar("Generator test loss", test_loss_mean)
        summary_writer.add_images("Generated test images", test_images[:MAX_SUMMARY_IMAGES])

    print("zavrsio")
    summary_writer.flush()

if __name__ == "__main__":
    test()