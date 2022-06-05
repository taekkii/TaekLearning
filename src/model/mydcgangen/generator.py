


# Generator for GAN

import torch
import torch.nn as nn
import torch.nn.functional as F
#from . import weightinit


LAST_CONV_SPATIAL = (4,4)

class Generator(nn.Module):
    
    def _get_block(self,in_channels,out_channels,kernel_size,stride=1,padding=0):
        i,o = in_channels,out_channels
        k,s,p = kernel_size,stride,padding

        return nn.ModuleList([nn.ConvTranspose2d(i,o,k,s,p,bias=False),
                              nn.BatchNorm2d(o),
                              nn.ELU(alpha=1.)])
                              
    def __init__(self,latent_dim,output_spatial=(3,32,32),last_layer_n_features=32):

        super().__init__()
        self.spatial = output_spatial
        self.noise_dim = latent_dim
        f = last_layer_n_features
        
        LH,LW = LAST_CONV_SPATIAL

        self.f = f

        c,h,w = output_spatial
    
        f *= (w//LW)

        self.layers = nn.ModuleList([
            *self._get_block(latent_dim,f,LW)
        ])
        ih,iw  =  LH,LW
        while ih < h  and  iw < w:
            self.layers.extend(self._get_block(f,f>>1,4,stride=2,padding=1))
            f>>=1
            ih<<=1
            iw<<=1
        self.layers.append(nn.ConvTranspose2d(f,c,1))

    def forward(self,z):
        x = z.reshape(-1,self.noise_dim,1,1)

        for layer in self.layers:
    #        print(x.shape)
            x = layer(x)
        return x.tanh()

    
if __name__ == '__main__':
    z = torch.rand(10)

    g = Generator(latent_dim=10,output_spatial=(3,128,128))

    x = g(z)
    print(x.shape)    