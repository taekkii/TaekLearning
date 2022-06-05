

import torch
import torch.nn as nn

from typing import Union, Optional


def distribute_by_sample( n:int , x:torch.Tensor , p_x:torch.Tensor ,eps = 1e-6):

    """
    given tensor of samples x and probability p(x), sample n new points following distribution~p(x)
    ARGUMENTS:
        n:
            # of new samples
        x:
            input of samples. batched(B x m) or not(m)
        p_x:
            p(x). Can be understood as a "weight" of likelihood of point x.
            Must be a same shape as x.
            if sum of tensor != 1, it will be automatically normalized to represent probability
        epsilon:
            recommend not to touch. Prevents divide by 0.
    RETURNS:
        [B x n] or [n] shaped tensor of sampled, ranging from 0 to 1
    """

    assert x.shape == p_x.shape, "x and p_x have different shapes"
    no_batch = x.dim()==1
    if no_batch: x,p_x = x.unsqueeze(0) , p_x.unsqueeze(0)
    assert x.dim()==2 , "the shape of tensors should be [B x m] or [m]"
    
    b,m = x.shape

    p_x += eps #prevents divide by 0
    cum_p_x = (p_x/p_x.sum(dim=-1).view(b,1)).cumsum(dim=-1) # if sum!=1, normalize first
    
    idx = torch.searchsorted(cum_p_x,torch.rand(b,n))
   
   
   
    def uniform_scatter(x,deviation):
        return x+torch.rand_like(x)*(2*deviation - deviation)



    ret = uniform_scatter( x.gather(1,idx) , 1/n).clamp(0.0,None)
    ret,_= ret.sort()

    return ret.squeeze() if no_batch \
      else ret



def sample_points( n_samples:int , 
                   ray_origin: torch.Tensor , 
                   ray_direction:torch.Tensor , 
                   t_sample:Optional[torch.Tensor] , 
                   density_t:Optional[torch.Tensor] , 
                   t_near=0. , 
                   t_far=1. ,
                   eps=1e-6):
    """
    given (a batch of) rays, sample (batch of) points
    ARGUMGNETS:
        n_samples:
            # of points/rays. (int)
        ray_origin:
            (batch of) origin of rays[3] OR [B x 3]
        ray_direction:
            (batch of) direction of rays[3] OR [B x 3]
        t_sample and density_t:
            if both are None, uniformly distributes (without randomness)
            if (batch of) tensor[B x m] is given, t_sample~density_t works as a distribution of t.
            if one of them are None, gives assertion error
        t_near/t_far: 
            lower/upper bound of t, where a single ray is represented by x=o+td
        
    RETURNS:
        points: (Batch of) sample points[ B x n_samples x 3]
        t: sampled t values[B x n_samples]
    """

    assert (t_sample==None and density_t==None)  or  (t_sample!=None and density_t!=None)

    o,d = ray_origin , ray_direction
    assert o.dim()==d.dim() and o.dim() in [1,2]
    if o.dim() == 1: o,d = o.unsqueeze(0),d.unsqueeze(0)
    assert o.shape[-1] == d.shape[-1] == 3

    
    n = n_samples
    
    b,m,_= ray_origin.shape
    if t_sample==None:
        t = torch.linspace(t_near+eps , t_far-eps , n)
    else:    
        t = distribute_by_sample(n , t_sample , density_t)
    t = t_near  +  (t_far - t_near)*t
    
    
    # t[b x n] , o[b x 3] , d[b x 3]
    # want[b x n x 3] : use broadcast rule
    points = o.view(b,1,3) + t.view(b,n,1)*d.view(b,1,3)
    return  points, t
    
    
def get_rgbsigma(model:nn.Module,points:torch.Tensor,minibatch_size):
    """
        run neural network on all points
        ARGUMENTS:
            model: NeRF Model
            points: [#ofpoints x 3] tensor
            minibatch_size: literally minibatch size
        RETURNS: (rgb,sigma) per each points [#ofpoints x 4]
    """
    
    return torch.stack([points[i:i+minibatch_size] for i in range(0 , points.shape[0] , minibatch_size)])
        

def render(model:nn.Module, ray_origin:torch.Tensor , ray_direction:torch.Tensor , n_samples=128 , minibatch_size=4096):
    """
    render with one network
    Arguments:
        model: 
            NeRF model. Use multinetwork_render() to render with more than two networks
        ray_origin:
            batch of origin of rays[b x 3]
        ray_direction:
            batch of direction of rays [b x 3]
    Returns:
        (Batch of)RGB of rendered tensor [b x 3]
    """

    b = ray_origin.shape[0]
    n = n_samples

    points,t = sample_points(n,ray_origin,ray_direction,None,None) #[b x Nsample x 3]
    
    rgb,sigma = torch.split(get_rgbsigma(model,points.view(-1,3),minibatch_size) , 3 , dim=-1)
    rgb,sigma = rgb.view(-1,n,3) , sigma.view(-1,n) # rgb:[b x n x 3], sigma[b x n]
    
    delta = torch.cat( [t[:,0].view(-1,1)  ,  t[ : , 1:] - t[: , :-1] ] , -1) #delta: [b x n]
    seq = sigma*delta #seq:[bxn]
    cseq = torch.cat( [ torch.ones(b).view(-1,1) , seq.cumsum(dim=-1)[ : , :-1] ] , -1) #cseq : [bxn]
    T = torch.exp(-cseq) # T[bxn]
    alpha = torch.ones_like(seq) - torch.exp(-seq) # alpha[bxn]
    res = (T*alpha*rgb.permute([2,0,1])).sum(dim=-1) #[bxn] * [bxn] * [3xbxn] = [3xbxn]. [3xbxn].sum(dim=-1) -> [3xb]
    
    return res.t() #[bx3]

    


