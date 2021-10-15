import torch as th 
import torch.nn as nn 

class DISBlock(nn.Module):
    def __init__(self, i_dim, n_dim, n_down):
        super(DISBlock, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(i_dim, n_dim, 4, 2, 1), 
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.body = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(n_dim * 2 ** i, n_dim * 2 ** (i + 1), 4, 2, 1), 
                nn.BatchNorm2d(n_dim * 2 ** (i + 1), 0.8), 
                nn.LeakyReLU(0.2)
            )
            for i in range(n_down)
        ])
        self.term = nn.Conv2d(n_dim * 2 ** n_down, 1, 3, 1, 1)
    
    def forward(self, X):
        return self.term(self.body(self.head(X)))

class Discriminator(nn.Module):
    def __init__(self, i_dim, n_dim, n_down, n_models):
        super(Discriminator, self).__init__()
        self.main_body = nn.ModuleList([])
        for _ in range(n_models):
            self.main_body.append(DISBlock(i_dim, n_dim, n_down))
        self.reducer = nn.AvgPool2d(i_dim, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, X):
        response = []
        for D in self.main_body:
            response.append(D(X))
            X = self.reducer(X)
        return response
    
    def mse(self, response, target):
        return sum([ th.mean((out - target) ** 2) for out in self.forward(response) ])


if __name__ == '__main__':
    dis = Discriminator(3, 64, 3, 3)
    print(dis)

    img = th.randn((1, 3, 128, 128))
    res = dis(img)
    print([ vec.shape for vec in res])
    