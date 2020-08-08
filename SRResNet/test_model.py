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
import PIL
from PIL import Image

from torch.autograd import Variable

from Modeli import *
from dataset import CelebaHQDataSet

import torch.nn as nn

import torch

IMG_SIZE_LR = 64    #velicine slika
IMG_SIZE_HR = 256
BATCH_SIZE = 16
NUM_WORKERS = 8
MAX_SUMMARY_IMAGES = 8
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


    checkpoint = torch.load("resnet_model_2try21.pt")
    generator.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']

    test_set = CelebaHQDataSet(IMG_SIZE_LR, IMG_SIZE_HR)
    test_dataloader = DataLoader(test_set,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=NUM_WORKERS,
                                drop_last=True)

    print("testiram")
    img = Image.open(r"C:\Users\psimluser\Desktop\Rezultati\Test1\vggmsegen.png")
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

            img1 = transforms.ToPILImage()(test_gen_hr[0].cpu())  #CPU
            img1.save(f"{i}gen.jpg")

            img2 = transforms.ToPILImage()(test_imgs_hr[0].cpu())  #CPU
            img2.save(f"{i}org.jpg")


            test_gen_features = vgg(test_gen_hr)
            test_real_features = vgg(test_imgs_hr)
            test_loss = MeanAbsErr(test_gen_features, test_real_features.detach())
            test_loss_sum += test_loss
            summary_writer.add_scalar("Generator test loss", test_loss_sum)
            summary_writer.add_images("Generated test images", test_gen_hr[:MAX_SUMMARY_IMAGES])
            summary_writer.add_images("Original lr test images", test_imgs_lr[:MAX_SUMMARY_IMAGES])
            summary_writer.add_images("Original hr test images", test_imgs_hr[:MAX_SUMMARY_IMAGES])
            if i == 0:
                break
            if i == rand_num:
                test_images = test_gen_hr

        #test_loss_mean = test_loss_sum / ((test_num + 1))
        #summary_writer.add_scalar("Generator test loss", test_loss_mean)
        #summary_writer.add_images("Generated test images", test_images[:MAX_SUMMARY_IMAGES])
        #summary_writer.add_images("Original test images", test_images[:MAX_SUMMARY_IMAGES])

    print("zavrsio")
    summary_writer.flush()

if __name__ == "__main__":
    test()