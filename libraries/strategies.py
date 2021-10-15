import cv2  

import numpy as np 
import operator as op 
import itertools as it, functools as ft 

import torch as th
import torchvision as tv 

from os import path 
from glob import glob 
from torchvision import transforms as T 
from tqdm import tqdm

from .log import logger 

def read_image(image_path, by='cv'):
    if by == 'cv':
        return cv2.imread(image_path, cv2.IMREAD_COLOR)
    if by == 'th':
        return tv.io.read_image(image_path)
    raise ValueError(by)

def th2cv(tensor_3d):
    red, green, blue = tensor_3d.numpy()
    return cv2.merge((blue, green, red))

def cv2th(bgr_image):
    blue, green, red = cv2.split(bgr_image)
    return th.from_numpy(np.stack([red, green, blue]))

def to_grid(batch_images, nb_rows=8, padding=10, normalize=True):
    grid_images = tv.utils.make_grid(batch_images, nrow=nb_rows, padding=padding, normalize=normalize)
    return grid_images

def denormalize(tensor_data, mean, std):
    mean = th.tensor(mean)
    std = th.tensor(std)
    return tensor_data * std[:, None, None] + mean[:, None, None]

def reparameterization(mu, logvar):
    std = th.exp(logvar / 2)
    sampled_z = th.as_tensor(np.random.normal(0, 1, mu.shape))
    z = sampled_z * std + mu
    return z

def save_image(img, path_to, nb_rows):
    img = img.cpu()
    merged_images = to_grid(img, nb_rows=nb_rows)
    rescaled_merged_images = th2cv(merged_images) * 255
    cv2.imwrite(path_to, rescaled_merged_images)

def sample_images(path_to, val_dataloader, generator, noise_dim):
    generator.eval()
    dvc = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    acc = []
    logger.debug('sampling process')
    idx = 0
    for img_A, img_B in tqdm(val_dataloader):
        real_A = img_A.repeat(noise_dim, 1, 1, 1).to(dvc)
        sampled_z = th.randn((noise_dim, noise_dim)).to(dvc)
        fake_B = generator(real_A, sampled_z)
        fake_B = th.cat([x for x in fake_B.data.cpu()], -1)[None, ...]
        img_sample = th.cat((img_A, img_B, fake_B), -1)
        acc.append(th.squeeze(img_sample))
        if len(acc) == 10:
            save_image(th.stack(acc), f'{path_to}_{idx:03d}.jpg', nb_rows=1)
            idx = idx + 1
            acc = []

    generator.train()
    