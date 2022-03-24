


import torch.nn as nn
import torch

# class SelfAttentionHead(nn.Module):
    

#     def __init__(self,dim,head_num):

#         super().__init__()
        
#         assert dim%head_num == 0
        
#         d = dim
#         dk = dim//head_num
        
#         self.d = d
#         self.dk_sqrt = dk**0.5

#         self.q_linear=nn.Linear(d,dk)
#         self.k_linear=nn.Linear(d,dk)
#         self.v_linear=nn.Linear(d,dk)
#         self.softmax = nn.Softmax(dim=-1)
#         #[2021/02/21] The Lab Leader's marvelousness is recorded here:

#         # Sanghyun's think: aggregate outputs from multi-head = correct!
#         # self.to_out = nn.Sequential(
#         #     nn.Linear(dk, dk),
#         # #    nn.Dropout(0.5)
#         # )
        

#     def forward(self,x):
#         q = self.q_linear(x) #q: Nxd_k
#         k = self.k_linear(x) #k: Nxd_k
#         v = self.v_linear(x) #v: Nxd_k
#         kt = k.transpose(-1,-2) #kt: d_k x N
#         z1 = torch.matmul(q,kt) / self.dk_sqrt # z1 : NxN
#         z2 = self.softmax(z1)   # z2 : NxN
#         z3 = torch.matmul(z2,v) # z3 : NxD_k
#         return z3


class MSA(nn.Module):

    def __init__(self,dim,head_num):
        super().__init__()
        self.Q = nn.Linear(dim,dim)
        self.K = nn.Linear(dim,dim)
        self.V = nn.Linear(dim,dim)
        self.softmax = nn.Softmax(dim=-1)
        self.O = nn.Linear(dim,dim)
        
        self.head = head_num
    
    def forward(self,x): #assume x = [B x 17 x 192] , h=12(head) for example
        # [2022/02/25->26] Note to self: [IMPORTANT] If you're doing something to do with "parallelization", 
        # You must somehow "matricize" your parallel input instead of using for loop
        # Bad example: torch.cat(*[ selfattention(x) for selfattention in modulelists] ])
        assert x.dim() == 3
        
        b,n,d = x.shape
        h = self.head
        scalar = (d//h)**0.5
        
        q = self.Q(x)    #       q = [B x 17 x 192] or [B x 17 x (12 16)]. so just changing the view does work.
        k = self.K(x)    #       k = [B x 17 x 192]
        v = self.V(x)    #       v = [B x 17 x 192]
                         #       let h = 12
        
        q = q.view(b,n,h,-1).transpose(1,2)        # q = [B x h x 17 x 16]
        k = k.view(b,n,h,-1).transpose(1,2)        # k = [B x h x 17 x 16]
        k = k.transpose(2,3)                       # k = [B x h x 16 x 17]
        
        attn_score = self.softmax(q@k / scalar)     # attn_score = [B x h x 17 x 17]
        
        v = v.view(b,n,h,-1).transpose(1,2)         # v = [B x h x 17 x 16]
        
        z = attn_score@v                            # z = [B x h x 17 x 16]
        z = z.transpose(1,2)                        # z = [B x 17 x h x 16]
        z = z.contiguous().view(b,n,-1)             # z = [B x 17 x 192]     
        return self.O(z)