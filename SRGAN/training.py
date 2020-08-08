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


EPOCH = 12

def train():
    cuda = torch.cuda.is_available()

    hr_shape = (IMG_SIZE_HR, IMG_SIZE_HR)

    discriminator = Discriminator(input_shape=(3, *hr_shape)) 
    generator = ResNetG()
    vgg = VGG()


    vgg.eval()

    MeanAbsErr = torch.nn.L1Loss()
    criterion_GAN = torch.nn.NLLLoss()


    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        vgg = vgg.cuda()
        MeanAbsErr = MeanAbsErr.cuda()
        criterion_GAN = criterion_GAN.cuda()



    optimizer = torch.optim.Adam(generator.parameters(), LR, betas=(ada1, ada2))
    optimizerD = torch.optim.Adam(discriminator.parameters(), LR, betas=(ada1, ada2))



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

    iters = 0
    for epoch in range(EPOCH):
        for i, img_batch in tqdm(enumerate(dataloader), total=total_iterations, desc=f"Epoch: {epoch}", unit="batches"):

            imgs_lr = img_batch[0].to(DEVICE)
            imgs_hr = img_batch[1].to(DEVICE)   #UCITAJ HIGH I LOW RESOLUTION IMAGES< TREBA DA BUDU NORMALIZOVANE NA INTERVALU 0 1

            global_step = epoch * total_iterations + i

            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            # ------------------
            #  Train Generators
            # ------------------

            optimizer.zero_grad()

            gen_hr = generator(imgs_lr)

            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

            #loss funkcija sadrzaja
            gen_features = vgg(gen_hr)
            real_features = vgg(imgs_hr)
            loss_content = MeanAbsErr(gen_features, real_features.detach())

            loss = (1/12.75) * loss_content + 0.003*loss_GAN #u rady pise da je skaliran izlaz iz vgg

            loss.backward()
            optimizer.step()

            
            optimizerD.zero_grad()

            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
                
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()
            #OVDE Treba neki logging staviti ima tqdm ili mozemo onaj tensor info ili tako nesto sto loguje na loalhost
            
             if global_step % 100 == 0:
                summary_writer.add_scalar("Generator loss", loss, global_step)
                summary_writer.add_scalar("Discriminator loss", loss_D, global_step)
                summary_writer.add_images("Generated images", gen_hr[:MAX_SUMMARY_IMAGES], global_step)
            
                
    
        torch.save({
            'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, f"SRGAN_generator_model_try{epoch}.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizerD.state_dict(),
            'loss': loss_D
        }, f"SRGAN_discriminator_model_try{epoch}.pt")

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
    
    summary_writer.flush()

if __name__ == "__main__":
    train()
