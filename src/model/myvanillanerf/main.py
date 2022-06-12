
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
pos_enc(x,L)
input : x [B x c]
output: positionally encoded x[Bx(2Lc)]
"""
def pos_enc(point:torch.Tensor , length:int):
    
    if length<=0: return point
    assert point.dim()==2

    p,L = point,length
    device = p.device

    b,ch = p.shape   
    # p : [b x c]

    geo_seq = ( 2. ** torch.arange(L,device=device) )# geo_seq: [L]
    seq = ( p.view(b,1,ch) * geo_seq.view(L,1) ).view(b,-1)# seq: [b x 1 x c]*[L x 1] = [b x L X c] --(view)-->[b x (Lc)]

    return torch.cat([p,torch.sin(seq) , torch.cos(seq)],dim=-1)


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
        

        pdim = self.pdim = pos_enc(torch.randn(1,3),Lx).shape[-1]
        ddim = self.ddim = pos_enc(torch.randn(1,3),Ld).shape[-1]

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

    def sigma_activation_fn(self,x):
        return 1.0+F.elu(x,alpha=1.)
    
    def forward(self,x,d):


        if x.dim()==1:x.unsqueeze(0)
        if d.dim()==1:x.unsqueeze(0)
        assert x.dim()==2 and d.dim()==2

        posenc_x = pos_enc(x , self.Lx)
        posenc_d = pos_enc(d , self.Ld)

        out = self.nonlinear( self.layer0(posenc_x) )
        
        
        for i,layer in enumerate(self.layers):
            if layer.is_skip_layer:
                out = self.nonlinear( layer(torch.cat([out,posenc_x],-1) ) )
            
            else:
                out = self.nonlinear( layer(out) )
                    


        sigma = self.sigma_activation_fn( self.layer_sigma(out) )
        out = self.layer1(out)
        out = self.layer2(torch.cat([out,posenc_d],-1))


        out = self.nonlinear(out)
        rgb = torch.sigmoid(self.layer3(out))

        return torch.cat([rgb,sigma],-1) 


if __name__ == '__main__':
    x = torch.tensor([[0.,1.,2.]])
    print(pos_enc(x,length=10).norm())
    
    # device="cuda:6"
    # x = torch.randn(4096*64,3).to(device)
    # d = torch.randn(4096*64,3).to(device)
    
    # import time
    # t0=time.time()


    # net = NeRF().to(device)
    
    # print(net(x,d)[:32,:])
    # print(f"{time.time()-t0:.3f}s")
