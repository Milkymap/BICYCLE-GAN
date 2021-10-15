import numpy as np 
import torch as th 

from libraries.log import logger 
from libraries.strategies import *  
from torch.utils.data import Dataset 

from torchvision import transforms as T 
from glob import glob 
from os import path 

class DHolder(Dataset):
    def __init__(self, root, extension='*.jpg', mapper=None, nb_items=None):
        if not path.isdir(root):
            logger.error(f'{root} is not a directory')
            raise ValueError(root)

        self.files = sorted(glob(path.join(root, extension)))
        if nb_items is not None:
            self.files = self.files[:nb_items]
        self.mapper = mapper 

    def normalize(self, image):
        normalized_image = image / th.max(image)  # value between 0 ~ 1 
        if self.mapper is not None: 
            mapped_image = self.mapper(normalized_image)
            return mapped_image
        return normalized_image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        current_file = self.files[idx]
        current_image = read_image(current_file, by='th')
        left_image, right_image = th.chunk(current_image, 2, dim=2)
        left_image = self.normalize(left_image)
        right_image = self.normalize(right_image)
        
        return left_image, right_image

if __name__ == '__main__':
    holder = DHolder('/home/ibrahima/Datasets/Edges2Shoes/train') 
    for i in range(10):
        left, right = holder[i]
        print(left.shape, right.shape)
        cv2.imshow('000', th2cv(left))
        cv2.imshow('001', th2cv(right))
        cv2.waitKey(0)
