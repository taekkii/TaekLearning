
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
pos_enc(x,L)
input : x [B x c]
output: positionally encoded x[Bx(2Lc)]
"""
def pos_enc(point:torch.Tensor , length:int):
    

    p,L = point,length
    
    if p.dim()==1: p = p.unsqueze(0)
    b,ch = p.shape
   
    # p : [b x c]
    assert p.dim()==2 

    # geo_seq: [L]
    geo_seq = torch.exp2( torch.arange(L) ) * torch.pi 

    # seq: [b x 1 x c]*[L x 1] = [b x L X c] --(view)-->[b x (Lc)]
    seq = ( p.unsqueeze(1) * geo_seq.unsqueeze(1) ).view(b,-1)
    
    # enc: [b x (2Lc)]
    enc = torch.zeros(b , seq.shape[-1]*2 )
    enc[:,0::2] = torch.sin(seq)
    enc[:,1::2] = torch.cos(seq)
    
    return enc
    
class NeRF(nn.Module):

    def __init__(self, in_channels=3, depth=10, width=256, posenc_Lx=10 , posenc_Ld=4 , skip_layers=[4] , nonlinear = F.relu ):
        
        super().__init__()
        ch = self.ch = in_channels
        d  = self.d  = depth
        w  = self.w  = width
        Lx = self.Lx = posenc_Lx
        Ld = self.Ld = posenc_Ld
        
        self.skip_layers = skip_layers
        self.nonlinear = nonlinear

        pdim = self.pdim = 2*ch*Lx
        ddim = self.ddim = 2*ch*Ld

        self.layer0  =  nn.Linear(pdim,w)
        self.layers  =  nn.ModuleList([])
        for i in range(1,d-3):
            if i in skip_layers : self.layers.append( nn.Linear(w+pdim , w))
            else                : self.layers.append( nn.Linear(w      , w))
            
            self.layers[-1].is_skip_layer = i in skip_layers

        self.layer1 = nn.Linear(w,w)
        self.layer_sigma = nn.Linear(w,1)
        
        self.layer2 = nn.Linear(w+ddim,w//2)
        self.layer3 = nn.Linear(w//2,3)

    def forward(self,x,d):
        
        if x.dim()==1:x.unsqueeze(0)
        if d.dim()==1:x.unsqueeze(0)
        assert x.dim()==2 and d.dim()==2
        
        posenc_x = pos_enc(x , self.Lx)
        posenc_d = pos_enc(d , self.Ld)
        

        out = self.nonlinear( self.layer0(posenc_x) )
        for layer in self.layers:
            if layer.is_skip_layer:
                out = self.nonlinear( layer(torch.cat([out,posenc_x],-1) ) )
            
            else:
                out = self.nonlinear( layer(out) )
        
        sigma = self.layer_sigma(out)
        out = self.layer1(out)
        out = self.layer2(torch.cat([out,posenc_d],-1))
        
        rgb = self.layer3(out)

        return torch.cat([rgb,sigma],-1)


if __name__ == '__main__':
     x = torch.randn(9,3)
     d = torch.randn(9,3)
     net = NeRF()
     net(x,d)

