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

LR_gen = 0.0001 #learning rate
LR_disc = 0.0001

EPOCH = 10

def train():
    summary_writer = SummaryWriter()
    cuda = torch.cuda.is_available()

    hr_shape = (IMG_SIZE_HR, IMG_SIZE_HR)

    discriminator = Discriminator(input_shape=(3, *hr_shape)) 
    #discriminator = Discriminator() 
    generator = ResNetG()
    vgg = VGG()


    vgg.eval()

    MeanAbsErr = torch.nn.L1Loss()
    #criterion_GAN = torch.nn.NLLLoss()
    criterion_GAN = torch.nn.BCELoss()


    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        vgg = vgg.cuda()
        MeanAbsErr = MeanAbsErr.cuda()
        criterion_GAN = criterion_GAN.cuda()



    optimizer = torch.optim.Adam(generator.parameters(), LR_gen, betas=(ada1, ada2))
    optimizerD = torch.optim.Adam(discriminator.parameters(), LR_disc, betas=(ada1, ada2))
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    checkpoint = torch.load("resnet_model_2try21.pt")
    generator.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch'] + 1
    #loss = checkpoint['loss']


    print("Setup finished")

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
    

    for epoch in range(EPOCH):
        for i, img_batch in tqdm(enumerate(train_dataloader), total=total_iterations, desc=f"Epoch: {epoch}", unit="batches"):

            imgs_lr = img_batch[0].to(DEVICE)
            imgs_hr = img_batch[1].to(DEVICE)   #UCITAJ HIGH I LOW RESOLUTION IMAGES< TREBA DA BUDU NORMALIZOVANE NA INTERVALU 0 1
            
            global_step = epoch * total_iterations + i

            #valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            #fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            valid = torch.ones(BATCH_SIZE, 1).to(DEVICE)
            fake = torch.zeros(BATCH_SIZE, 1).to(DEVICE)
            # ------------------
            #  Train Generators
            # ------------------

            optimizer.zero_grad()

            gen_hr = generator(imgs_lr)

            if i % 2 == 1:
                #print(discriminator(gen_hr).size())
                loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

                #loss funkcija sadrzaja
                gen_features = vgg(gen_hr)
                real_features = vgg(imgs_hr)
                loss_content = MeanAbsErr(gen_features, real_features.detach())

                loss = (1/12.75) * loss_content + 0.001*loss_GAN #u rady pise da je skaliran izlaz iz vgg
                loss.backward()
                optimizer.step()

                if i % 100 == 1:
                    summary_writer.add_scalar("Generator loss", loss, global_step)
            #Disriminator

            else:
                optimizerD.zero_grad()

                #loss_real = criterion_GAN(discriminator(imgs_hr), valid)
                #loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

                disc_real = discriminator(imgs_hr)
                #print(f"Discriminator on real: {disc_real}")
                disc_fake = discriminator(gen_hr.detach())
                #print(f"Discriminator on fake: {disc_fake}")

                loss_real = criterion_GAN(disc_real, valid)
                loss_fake = criterion_GAN(disc_fake, fake)

                #print(f"Loss real: {loss_real}")
                #print(f"Loss fake: {loss_fake}")
                    
                loss_D = (loss_real + loss_fake) / 2
                LOSS = loss_D

                loss_D.backward()
                optimizerD.step()
                #OVDE Treba neki logging staviti ima tqdm ili mozemo onaj tensor info ili tako nesto sto loguje na loalhost
                
                if i % 100 == 0:
                    summary_writer.add_scalar("Discriminator loss", loss_D, global_step)
                    summary_writer.add_scalar("Discriminator real loss", loss_real, global_step)
                    summary_writer.add_scalar("Discriminator fake loss", loss_fake, global_step)
                    summary_writer.add_images("Generated images", gen_hr[:MAX_SUMMARY_IMAGES], global_step)
        torch.save({
             'epoch': epoch,
             'model_state_dict': generator.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'loss': loss
        }, f"SRGAN_generator_model{epoch}.pt")    
                
    
    torch.save({
        'epoch': 10,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, f"SRGAN_generator_model{epoch}.pt")

    torch.save({
        'epoch': 10,
        'model_state_dict': discriminator.state_dict(),
        'optimizer_state_dict': optimizerD.state_dict(),
    }, f"SRGAN_discriminator_model{epoch}.pt")

    
    summary_writer.flush()

if __name__ == "__main__":
    train()
