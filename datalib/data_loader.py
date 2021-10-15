import torch as th 
import torch.nn as nn 
from torch.utils.data import DataLoader 

class DLoader:
    def __init__(self, dataset, shuffle, batch_size, sampler=None):
        self.core = DataLoader(
			dataset=dataset, 
			shuffle=shuffle, 
			batch_size=batch_size, 
			drop_last=True, 
			sampler=sampler 
		)

if __name__ == '__main__':
    pass 
