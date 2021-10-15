import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

from torchvision.models import resnet18 as R18 

class Extractor(nn.Module):
    def __init__(self, noise_dim, path2res18=None):
        super(Extractor, self).__init__()
        if path2res18 is None:
            resnet18 = R18(pretrained=False)
        else:
            resnet18 = th.load(path2res18, map_location='cpu')
        self.extractor = nn.Sequential(*list(resnet18.children())[:-3])
        self.reducer = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        
        self.linear0 = nn.Linear(256, noise_dim)
        self.linear1 = nn.Linear(256, noise_dim)

    def forward(self, img):
        out = self.extractor(img)
        out = self.reducer(out)
        out = out.view(out.size(0), -1)
        mu = self.linear0(out)
        logvar = self.linear1(out)
        return mu, logvar

if __name__ == '__main__':
    ext = Extractor(100)
    print(ext)