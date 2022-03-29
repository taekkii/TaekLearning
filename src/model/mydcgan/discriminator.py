
import torch.nn as nn
#from . import weightinit

LAST_CONV_SPATIAL = (4,4)

class Discriminator(nn.Module):
    def _get_block(self,in_channels,out_channels,kernel_size,stride=1,padding=0, dim=1):
        i,o = in_channels, out_channels
        k,s,p = kernel_size , stride , padding

        return nn.ModuleList([
            nn.Conv2d(i,o,k,s,p,bias=False),
            nn.LeakyReLU(0.2),
        ])
        
    def __init__(self,first_layer_n_features=32,spatial=(3,32,32)):
        
        super().__init__()
        c,h,w = spatial
        LH,LW = LAST_CONV_SPATIAL

        f = first_layer_n_features

        self.spatial = spatial
  
        self.layers = self._get_block(c,f,1)
        while h>LH  and  w>LW:
            self.layers.extend(self._get_block(f,f<<1,4,stride=2,padding=1))
            h>>=1
            w>>=1
            f<<=1
        self.layers.append(nn.Conv2d(f,1,LW,stride=1,padding=0,bias=False))
        self.layers.append(nn.Sigmoid())

#        self.apply(weightinit.weight_init)


    def forward(self,x):
        for layer in self.layers:
   #         print(x.shape)
            x = layer(x)
        return x.view(-1,1)

if __name__ == "__main__":
    import torch;x = torch.randn(2,3,128,128)
    net = Discriminator(spatial=(3,128,128))
    y=net(x)
    print(y.shape)
