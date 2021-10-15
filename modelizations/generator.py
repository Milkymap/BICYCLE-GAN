import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

import itertools as it, functools as ft 

class DWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super(DWBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels) if normalize else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, X):
        return self.body(X)

class UPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, last_output=False):
        super(UPBlock, self).__init__()
        self.head = nn.Upsample(scale_factor=2)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU() if not last_output else nn.Tanh()
        )

    def forward(self, X, S=None):
        U = self.head(X)
        if S is not None:
            US = th.cat([U, S], dim=1)
            return self.body(US)
        return self.body(U)

class Generator(nn.Module):
    def __init__(self, shape, noise_dim):
        super(Generator, self).__init__()
        channels, height, width = shape  
        self.linear = nn.Linear(noise_dim, height * width)
        self.encode = nn.ModuleList([
            DWBlock(channels + 1,  64),  # 0128x064
            DWBlock(64,  128),           # 0256x064
            DWBlock(128, 256),           # 0512x128
            DWBlock(256, 512),           # 1024x256
            DWBlock(512, 512),           # 1024x512
            DWBlock(512, 512),           # 1024x512
            DWBlock(512, 512, False)
        ])
        self.decode = nn.ModuleList([
            UPBlock(1024, 512),
            UPBlock(1024, 512),
            UPBlock(1024, 256),
            UPBlock( 512, 128),
            UPBlock( 256,  64),
            UPBlock( 128,  64),
            UPBlock(  64, channels, True),
        ])
    
    def forward(self, X, Z):
        B, _, H, W = X.shape
        A = self.linear(Z)
        A = th.reshape(A, (B, 1, H, W))
        AZ = th.cat([X, A], dim=1)
        R = []
        for E in self.encode:
            AZ = E(AZ)
            R.append(AZ)

        I = len(R) - 1
        C = R[I] 
        for D in self.decode:
            if I > 0:
                C = D(C, R[I - 1])
                I = I - 1 
            else:
                C = D(C, None)
        return C

if __name__ == '__main__':
    dvc = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    gen = Generator((3, 128, 128), 100)
    gen = gen.to(dvc)
    print(gen)
    
    for _ in range(100):
        img = th.randn((2, 3, 128, 128))
        vec = th.randn((2, 100))
        res = gen(img.to(dvc), vec.to(dvc))
        print(res.shape)