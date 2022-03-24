


import torch
import torch.nn as nn
from .msa import MSA

class TransformerEncoder(nn.Module):
    
    def __init__(self,dim,head_num,mlp_size):
        
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(dim,eps=1e-6)
        self.msa = MSA(dim,head_num)
        self.layer_norm2 = nn.LayerNorm(dim,eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim,mlp_size),
            nn.GELU(),
   #         nn.LayerNorm(mlp_size),#experiment
            nn.Linear(mlp_size,dim),
    #        nn.GELU(), # exp : failure
     #       nn.LayerNorm(dim),#experiment
        )



    def forward(self,x):  #suppose x: n x d
        x = self.msa(self.layer_norm1(x)) + x
        x = self.mlp(self.layer_norm2(x)) + x
        return x
