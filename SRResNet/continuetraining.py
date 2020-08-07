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


project_path = os.path.dirname(os.path.dirname(__file__))
project_path = os.path.abspath(project_path)

IMG_SIZE_LR = 64    #velicine slika
IMG_SIZE_HR = 256
BATCH_SIZE = 16
NUM_WORKERS = 8
MAX_SUMMARY_IMAGES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ada1 = 0.9          #koeficijenti za adam optimizer
ada2 = 0.999

LR = 0.0002 #learning rate

START_EPOCH = 
EPOCH = 12

generator = ResNetG()
generator.load_state_dict(torch.load(#put))
generator.eval()


def compute_total_variation_loss(img, weight):      
        tv_h = ((img[:,:,1:,:] - img[:,:,:-1,:]).pow(2)).sum()
        tv_w = ((img[:,:,:,1:] - img[:,:,:,:-1]).pow(2)).sum()    
        return weight * (tv_h + tv_w)

def train():
    summary_writer = SummaryWriter()
    cuda = torch.cuda.is_available()
    print(os.path.join(project_path, "Checkpoints"))
    hr_shape = (IMG_SIZE_HR, IMG_SIZE_HR)


    generator = ResNetG()
    vgg = VGG()
    #generator.to(DEVICE)
    #vgg.to(DEVICE)

    vgg.eval()

    MeanAbsErr = torch.nn.L1Loss()

    if cuda:
        generator = generator.cuda()
        vgg = vgg.cuda()
        MeanAbsErr = MeanAbsErr.cuda()





    optimizer = torch.optim.Adam(generator.parameters(), LR, betas=(ada1, ada2))


    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    dataset = CelebaHQDataSet(IMG_SIZE_LR, IMG_SIZE_HR)
    train_set, valid_set = random_split(dataset, [28900, 100])
    train_dataloader = DataLoader(train_set,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=NUM_WORKERS,
                            drop_last=True)
    valid_dataloader = DataLoader(valid_set,
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

    total_iterations = 28900 // BATCH_SIZE

    for epoch in range(START_EPOCH,EPOCH):
        generator.train()
        for i, img_batch in tqdm(enumerate(train_dataloader), total=total_iterations, desc=f"Epoch: {epoch}", unit="batches"):
            
            global_step = epoch * total_iterations + i

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
            var_loss = compute_total_variation_loss(gen_hr, 1)
            loss = loss_content + var_loss

            loss.backward()
            optimizer.step()

            if global_step % 100 == 0:
                summary_writer.add_scalar("Generator loss", loss_content, global_step)
                summary_writer.add_images("Generated images", gen_hr[:MAX_SUMMARY_IMAGES], global_step)
            
                
    
        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, f"resnet_model_try{epoch}.pt")
        generator.eval()
        with torch.no_grad():
            valid_loss_sum = 0
            valid_num = 0
            valid_images = []
            rand_num = random.randint(0,5)
            for i, valid_img_batch in enumerate(valid_dataloader):
                valid_num = i
                valid_imgs_lr = valid_img_batch[0].to(DEVICE)
                valid_imgs_hr = valid_img_batch[1].to(DEVICE) 
                valid_gen_hr = generator(valid_imgs_lr)
                valid_gen_features = vgg(valid_gen_hr)
                valid_real_features = vgg(valid_imgs_hr)
                valid_loss = MeanAbsErr(valid_gen_features, valid_real_features.detach())
                valid_loss_sum += valid_loss
                if i == rand_num:
                    valid_images = valid_gen_hr
            valid_loss_mean = valid_loss_sum / ((valid_num + 1))
            summary_writer.add_scalar("Generator validation loss", valid_loss_mean, global_step)
            summary_writer.add_images("Generated validation images", valid_images[:MAX_SUMMARY_IMAGES], global_step)


        #OVDE Treba neki logging staviti ima tqdm ili mozemo onaj tensor info ili tako nesto sto loguje na loalhost
    
    summary_writer.flush()
if __name__ == "__main__":
    train()

