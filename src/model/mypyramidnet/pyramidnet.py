

import torch.nn as nn
import torch.nn.functional as F

class PyramidShortcut(nn.Module):
    def __init__(self,in_channels,out_channels,downsample=False):
        super().__init__()

        self.additional = out_channels-in_channels
        self.pool = nn.AvgPool2d(kernel_size=2,stride=2) if downsample else nn.Identity()
    
    def forward(self,x):
        x = F.pad(x,(0,0,0,0,0,self.additional))
        return self.pool(x)

class PyramidBottleneckBlock(nn.Module):
    pass ## later..

class PyramidBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        i,o = in_channels,out_channels
        self.f = nn.Sequential(
            nn.BatchNorm2d(i),
            nn.Conv2d(i,o,3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(),
            nn.Conv2d(o,o,3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(o)
        )
        self.shortcut = PyramidShortcut(i,o,downsample=stride>1)
    
    def forward(self,x):
        return self.f(x) + self.shortcut(x)

class PyramidNet(nn.Module):
    def __init__(self,in_channels,n_classes,n,alpha,block=PyramidBlock,spatial=32):
        
        super().__init__()
        self.depth = n*(6 if block==PyramidBlock else 9)+2
        
        print(':'*5,"Taek's pyramidnet",':'*5,end=' ')
        print(f"alpha={alpha} , depth={self.depth}")
        print('\n')

        ch=16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,ch,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch)
        )
        additional = alpha/(3*n)

        layers=[]
        for i in range(3*n):
            stride = 2 if i>0 and i%n==0 else 1
            layers.append( block(int(round(ch)),int(round(ch+additional)),stride) )
            ch+=additional
        
        self.convx = nn.Sequential(*layers)
        
        ch = int(round(ch))
        
        self.tail_layer = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.AvgPool2d(spatial//4,stride=1),
            nn.Flatten()
        )
        self.fc = nn.Linear(ch,n_classes)
    
    def forward(self,x):
        z1 = self.conv1(x)
        z2 = self.convx(z1)
        z3 = self.tail_layer(z2)
        z4 = self.fc(z3)
        return z4

def get_pyramidnet_for_cifar(alpha,depth=110,bottleneck = False):
    if bottleneck:
        block  = PyramidBottleneckBlock
        factor = 9
    else:
        block  = PyramidBlock
        factor = 6
    
    n=(depth-2)//factor
    return PyramidNet(3,10,n,alpha,block)
