

import torch.nn as nn
import torch

import math

from .encoder import TransformerEncoder

class PositionalEncoding(nn.Module): ### Reference: pytorch official
    
    sinusodial_encoding = False

    def __init__(self,n,d):
        super().__init__()
        self.n , self.d = n,d
        if PositionalEncoding.sinusodial_encoding:
            pos = torch.arange(n).unsqueeze(1)
            div = torch.exp(torch.arange(0,d,2) * (-math.log(10000.) / d) ) 
            pos_encoding = torch.randn(n,d)
            pos_encoding[: , 0::2] = torch.sin(pos*div)
            pos_encoding[: , 1::2] = torch.cos(pos*div)
            self.pe = nn.Parameter(pos_encoding.unsqueeze(0))
            #self.register_buffer('pe',pos_encoding)
        else:
            pos_encoding = nn.Parameter(torch.randn(1,n,d) )
            self.pe = pos_encoding
        
        
    def forward(self,x):                         # x : [B x 17 x D]
        return x + self.pe
    
class Transformer(nn.Module):
    def __init__(self,n_channels,n_classes,patch_size,D,num_heads,num_encoders,mlp_size,spatial=32):
        super().__init__()

        p,c = patch_size,n_channels

        self.patch_size = p
        self.n_channels = c

        n = (spatial//patch_size)**2
        #self.to_patch = nn.Unfold(kernel_size=p,stride=p)
        self.to_patch = nn.Conv2d(n_channels,D,kernel_size=patch_size,stride=patch_size)
    #     self.projection = nn.Sequential(
    #  #       nn.LayerNorm(p**2*c), ##LN is not part of the paper, but works better by experiment
    #         nn.Linear(p**2 * c, D),
    #  #       nn.LayerNorm(D)
    #     )

        self.x_class    = nn.Parameter(data=torch.randn(1,1,D)) ### [2021/02/21] init should be considered
     
        self.positional_encode = PositionalEncoding(n+1,D) 
        self.encoders = nn.ModuleList(
            [TransformerEncoder(dim=D,
                                head_num=num_heads,
                                mlp_size=mlp_size) for _ in range(num_encoders)]
        )
        self.layer_norm = nn.LayerNorm(D,eps=1e-6)
        self.fc = nn.Linear(D,n_classes) 

       
        #self.init_weights()
        

    def forward(self,x):                                            # x             : [B x 3 x 32 x 32], patch-size:8 for example..
        assert x.dim()==3 or x.dim()==4
        if x.dim()==3: x.unsqueeze(0)
        b = x.shape[0]
        
        x = self.to_patch(x).flatten(2).transpose(-1,-2)
       # x = self.projection(x)
        # x_patches = self.to_patch(x).transpose(-1,-2)               # x_patches     : [Bx16x 192]
        # x_proj    = self.projection(x_patches)                      # x_proj        : [Bx16x D]
        
        
        x_class   = self.x_class.expand(b,-1,-1)                    # x_class: [Bx 1 x D] 
                                                                    # (B shallow copies of [1xD])
       # timestamp()
        x = torch.cat((x_class,x),dim=1)                          # x : [Bx17xD]
        x = self.positional_encode(x)                             # x : [BX17xD]

        for encoder in self.encoders:
            x = encoder(x)
   
        x = x                                              # x : [BxD]

        x = self.fc(self.layer_norm(x)[:,0] )                          # x : [Bx10]'s 0th element
  
        return x
  

if __name__ == '__main__':
   # net = Transformer(image_size=8,patch_size=2,num_classes=10,dim=16,depth=7,heads=8,mlp_dim=384*4)
    net = Transformer(n_channels=3,n_classes=10,patch_size=2,D=16,num_heads=8,num_encoders=7,mlp_size=384*4,spatial=8)
    x = torch.arange(2*3*8*8).float().view(2,3,8,8)
    net(x)
