import os 
import click
import pickle as pk 
import numpy as np 

import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

from torchvision import transforms as T
from torchvision.io import image 

from datalib.data_holder import DHolder
from datalib.data_loader import DLoader

from libraries.log import logger 
from libraries.strategies import *
from modelizations.extractor import Extractor 
from modelizations.generator import Generator
from modelizations.discriminator import Discriminator

from config import * 

def main():
    logger.debug('... TRAINING PROCESS ...')

    transformer = T.Compose([
        T.Resize((IMG_WIDTH, IMG_HEIGHT)),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    data_holder = DHolder(IMAGES_PATH, '*.jpg', transformer)
    data_loader = DLoader(data_holder, True, BATCH_SIZE, None)
    
    eval_holder = DHolder(VALIDATION, '*.jpg', transformer)
    eval_loader = DLoader(eval_holder, False, 1, None)

    nb_images = len(data_holder)
    
    logger.debug('datalib initialization complete 100%')
    logger.debug(f'cuda status : {th.cuda.is_available()}')
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

    if USE_PRETRAINED:
        logger.debug('pretrained network will be used')
        ext_network = Extractor(NOISE_DIM, path.join(MODELS_PATH, MODELS_NAME)).to(device)
        gen_network = Generator((3, IMG_HEIGHT, IMG_WIDTH), NOISE_DIM).to(device)
        clr_dis_network = Discriminator(3, 64, 3, 3).to(device)
        vae_dis_network = Discriminator(3, 64, 3, 3).to(device)

        ext_network.load_state_dict(th.load(path.join(MODELS_DUMP, f'ext_network_{START_EPOCH:03d}.pth')))
        gen_network.load_state_dict(th.load(path.join(MODELS_DUMP, f'gen_network_{START_EPOCH:03d}.pth')))
        clr_dis_network.load_state_dict(th.load(path.join(MODELS_DUMP, f'clr_dis_network_{START_EPOCH:03d}.pth')))
        vae_dis_network.load_state_dict(th.load(path.join(MODELS_DUMP, f'vae_dis_network_{START_EPOCH:03d}.pth')))
    else:
        ext_network = Extractor(NOISE_DIM, path.join(MODELS_PATH, MODELS_NAME)).to(device)
        gen_network = Generator((3, IMG_HEIGHT, IMG_WIDTH), NOISE_DIM).to(device)
        clr_dis_network = Discriminator(3, 64, 3, 3).to(device)
        vae_dis_network = Discriminator(3, 64, 3, 3).to(device)
        
    ext_optimizer = th.optim.Adam(ext_network.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))
    gen_optimizer = th.optim.Adam(gen_network.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))
    clr_dis_optimizer = th.optim.Adam(clr_dis_network.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))
    vae_dis_optimizer = th.optim.Adam(vae_dis_network.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))

    logger.debug(f'all networks are loaded on {device}')
    mae_loss = nn.L1Loss()

    epoch_counter = START_EPOCH
    while epoch_counter < NB_EPOCHS:
        images_counter = 0
        for XA, XB in data_loader.core:
            XA = XA.to(device)
            XB = XB.to(device)
            
            RL = th.ones(1).to(device)
            FL = th.zeros(1).to(device)

            # -------------------------
            # TRAIN GENERATOR & ENCODER
            # -------------------------

            gen_optimizer.zero_grad()
            ext_optimizer.zero_grad()

            MU, LV = ext_network(XB)
            Z0 = reparameterization(MU.cpu(), LV.cpu()).float().to(device)
            B0 = gen_network(XA, Z0)

            # Pixelwise loss of translated image by VAE
            loss_pixel = mae_loss(B0, XB)
            # Kullback-Leibler divergence of encoded B
            loss_kl = 0.5 * th.sum(th.exp(LV) + MU ** 2 - LV - 1)
            # Adversarial loss
            loss_VAE_GAN = vae_dis_network.mse(B0, RL)
            
            # TRAIN CLR-GAN
            Z1 = th.randn((XB.shape[0], NOISE_DIM)).to(device)
            B1 = gen_network(XA, Z1)
            loss_LR_GAN = clr_dis_network.mse(B1, RL)

            # ----------------------------------
            # Total Loss (Generator + Encoder)
            # ----------------------------------

            loss_GE = loss_VAE_GAN + loss_LR_GAN + LAMBDA_PIXEL * loss_pixel + LAMBDA_KL * loss_kl

            loss_GE.backward(retain_graph=True)
            ext_optimizer.step()

            # ---------------------
            # Generator Only Loss
            # ---------------------

            # Latent L1 loss
            MU, LV = ext_network(B1)
            loss_latent = LAMBDA_LATENT * mae_loss(MU, Z1)

            loss_latent.backward()
            gen_optimizer.step()

            # ----------------------------------
            #  Train Discriminator (cVAE-GAN)
            # ----------------------------------

            vae_dis_optimizer.zero_grad()

            loss_VAE = vae_dis_network.mse(XB, RL) + vae_dis_network.mse(B0.detach(), FL)

            loss_VAE.backward()
            vae_dis_optimizer.step()

            # ---------------------------------
            #  Train Discriminator (cLR-GAN)
            # ---------------------------------

            clr_dis_optimizer.zero_grad()

            loss_CLR = clr_dis_network.mse(XB, RL) + clr_dis_network.mse(B1.detach(), FL)

            loss_CLR.backward()
            clr_dis_optimizer.step()

            logger.debug(f'[{epoch_counter:03d}/{NB_EPOCHS}] [{images_counter:04d}/{nb_images}] CLR_Loss: {loss_CLR:7.3f} VAE_Loss: {loss_VAE:7.3f} GEN_Loss: {loss_GE:7.3f} | {gen_network.training}')
            if images_counter % SNAPSHOT_INTERVAL == 0:
                sample_images(f'{FAKE_IMAGES_STORAGE}/image{epoch_counter:03d}_{images_counter:05d}', eval_loader.core, gen_network, NOISE_DIM)
                print('')

            images_counter += XA.shape[0]

        if epoch_counter % CHECKPOINT == 0:
            th.save(ext_network.state_dict(), f'{MODELS_DUMP}/ext_network_{epoch_counter:03d}.pth')
            th.save(gen_network.state_dict(), f'{MODELS_DUMP}/gen_network_{epoch_counter:03d}.pth')
            th.save(vae_dis_network.state_dict(), f'{MODELS_DUMP}/vae_dis_network_{epoch_counter:03d}.pth')
            th.save(clr_dis_network.state_dict(), f'{MODELS_DUMP}/clr_dis_network_{epoch_counter:03d}.pth')

            logger.success('checkpoint reached, models was saved ...!')

        epoch_counter += 1
    # end training while loop

if __name__ == '__main__':
    main()
