

import torch
import torch.nn as nn

from typing import Union, Optional

import utils.debug as dbg
from . import ray
import tqdm
@torch.no_grad()
def distribute_by_sample( n:int , p_x:torch.Tensor ,eps=1e-6):

    """
    given tensor of samples x and probability p(x), sample n new points following distribution~p(x)
    ARGUMENTS:
        n:
            # of new samples
        p_x:
            [B x n]shaped tensor
            p(x). Can be understood as a "weight" of likelihood of point x~U(0,1).
            Must be a same shape as x.
            if sum of tensor != 1, it will be automatically normalized to represent probability
        epsilon:
            recommend not to touch. Prevents divide by 0.
    RETURNS:
        [B x n] shaped tensor of sampled, ranging from 0 to 1
    """

    b,m = p_x.shape
    device=p_x.device

    #prevents divide by 0
    cum_p_x = ((p_x+eps)/(p_x+eps).sum(dim=-1).view(b,1)).cumsum(dim=-1) # if sum!=1, normalize first
    
    idx = torch.searchsorted(cum_p_x,torch.rand(b,n,device=device)).clamp(0,m-1)
   
   
    def uniform_scatter(x,deviation):
        return x+torch.rand_like(x)*(2*deviation - deviation)


    ret = uniform_scatter( torch.linspace(eps,1-eps,m,device=device).view(1,-1).expand([b,-1]).gather(1,idx) , 1/n).clamp(0.0,None)
    ret,_= ret.sort()
    return ret


@torch.no_grad()
def sample_points( n_samples:int , 
                   ray_origin: torch.Tensor , 
                   ray_direction:torch.Tensor , 
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
        density_t:
            density given t~U(0,1), sampled [n_sample] times.
        t_near/t_far: 
            lower/upper bound of t, where a single ray is represented by x=o+td
        
    RETURNS:
        points: (Batch of) sample points[ B x n_samples x 3]
        t: sampled t values[B x n_samples]
    """


    o,d = ray_origin , ray_direction
    assert o.dim()==d.dim() and o.dim() in [1,2]
    if o.dim() == 1: o,d = o.unsqueeze(0),d.unsqueeze(0)
    assert o.shape[-1] == d.shape[-1] == 3 , "o.shape = {} , d.shape={}".format(o.shape,d.shape)

    
    n = n_samples
    
    b,_= ray_origin.shape
    
    if density_t==None:
        t_raw = torch.linspace(eps , 1-eps , n , device=o.device).view(1,-1).expand([b,-1])
    else:
        t_raw = distribute_by_sample(n , density_t)
    t = t_near  +  (t_far - t_near)*t_raw
    
    
    # t[b x n] , o[b x 3] , d[b x 3]
    # want[b x n x 3] : use broadcast rule
    
    points = o.view(b,1,3) + t.view(b,n,1)*d.view(b,1,3)
    return  points, t
    


def get_rgbsigma(model:nn.Module,points:torch.Tensor, directions:torch.Tensor, minibatch_size:int=1<<20):
    """
        run neural network on all points
        ARGUMENTS:
            model: NeRF Model
            points: [#ofpoints x 3] tensor
            minibatch_size: literally minibatch size
        RETURNS: (rgb,sigma) per each points [#ofpoints x 4]

        [Note] Why need this?
            # of sample points > # of sample rays, so we have no choice but using "for loop" for now
    """
    return torch.cat([model(points[i:i+minibatch_size],directions[i:i+minibatch_size]) for i in range(0 , points.shape[0] , minibatch_size)],dim=0)
       
def integral_rgb(rgb,sigma,t):
    """
    Performs volumetric rendering and returns rgb-map.
    [ARGUMENTS]
        rgb:   [#rays x #points/ray x 3] rgb vals
        sigma: [#rays x #points/ray]     density vals
        t:     [#rays x #points/ray]     depth distribution/rays. cf) o+"t"d.
                                         (can be regarded as z,       o+"z"d)
    [RETURNS]
        rgb_map: [#rays x 3]  rendered rgb values per rays.
        weight : [#rays x #points/ray] cumulative weight per each point samples
    """
    b,n,_ = rgb.shape
    device = rgb.device

    delta = torch.cat( [t[ : , 1:] - t[: , :-1]  ,torch.ones(b,1,device=device)*1e10 ] , -1) #delta --(cat[bx(n-1)],[bx1])--> [b x n]
    seq = sigma*delta #seq:[bxn]
    
    alpha = torch.ones_like(seq) - torch.exp(-seq) # alpha[bxn]
    T = torch.cumprod(  torch.cat([ torch.ones(b,1,device=device) , (1.-alpha + 1e-6)[:,:-1] ] ,-1) , dim=-1 ) #T(roughly,=cumprod(1-alpha)) :[b x n]
    weight = alpha*T # weight: [bxn]

    res = (weight.view(b,n,1) * rgb).sum(dim=-2) #[bxnx1] * [bxnx3] = [bxnx3]. [bxnx3].sum(dim=-2) -> [bx3]
    return res,weight

# [credit]yenchenlin nerf-pytorch
# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    device=bins.device

    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples,device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples],device=device)

   

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
# [credit]yenchenlin nerf-pytorch
def raw2outputs(rgb,sigma, z_vals, rays_d):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists)
    device=rgb.device
    b,n,_ = rgb.shape
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.ones(b,1,device=device)*1e10 ], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
  
    alpha = raw2alpha(sigma , dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1),device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
   
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]


    return rgb_map,weights

def render( model_coarse:nn.Module,
            model_fine:Optional[nn.Module],
            ray_o:torch.Tensor, 
            ray_d:torch.Tensor, 
            n_samples_coarse=64 , 
            n_samples_fine=0 , 
            t_near=1e-6,
            t_far=1.0,
            coarse_ratio=None):
    """
    hierarchical render with a coarse network and a fine network
    Arguments:
        model_coarse:nn.Module,
            Coarse network(or just a single nerf if you don't want hierarcical rendering)

        model_fine:Optional[nn.Module],
            Fine network(or just pass None if you want one-network rendering)

        ray_o:torch.Tensor/ray_d:torch.Tensor: rays[#RAYS x 3] each

        n_samples_coarse:
            #of sample points per rays for "coarse" network
        n_samples_fine:
            #of sample points per rays for "fine" network

        t_near/t_far: min/max depth. (consider t as z)
        
        coarse_ratio=0.5:
            mixture constant c for final rgb value.
            i.e)final_rgb = c*rgb_coarse + (1-c)*rgb_fine
            IF THIS IS NOT GIVEN: just returns seperate RGBs from coarse and fine network. 
    Returns:
        (Batch of)RGB of rendered tensor [b x 3]
        (sample point depth distribution vector per rays) t(or z, whatever you're accustomed to) [b x n_samples]
    """
    b = ray_o.shape[0]
    n_c,n_f = n_samples_coarse,n_samples_fine
    
    
    #==== [COARSE] ====#
    points_c,t_c = sample_points(n_c,ray_o,ray_d,
                                 density_t=None,
                                 t_near=t_near,
                                 t_far=t_far) #[b x Nsample x 3]
  
  
    rgb_c,sigma_c = torch.split(get_rgbsigma(model_coarse,points_c.view(-1,3),ray_d.view(b,1,3).expand([b,n_c,3]).reshape(-1,3)) , 3 , dim=-1)
    rgb_c,sigma_c = rgb_c.view(-1,n_c,3) , sigma_c.view(-1,n_c) # rgb:[b x n x 3], sigma[b x n]


    rgb_map_c,weight_c = integral_rgb(rgb_c,sigma_c,t_c)  #[bx3]
    #rgb_map_c,weight_c = raw2outputs(rgb_c,sigma_c,t_c,ray_d)
  
    if model_fine is None or n_samples_fine<=0:
        if coarse_ratio is None: return rgb_map_c,None
        else: return rgb_map_c
    #==== [FINE] ====#
    points_f,t_f = sample_points(n_f , ray_o , ray_d , 
                                 density_t=weight_c.detach(),
                                 t_near=t_near,
                                 t_far=t_far) #[b x Nsample x 3]
    # t_f,_ = sample_pdf( (t_c[:,1:] + t_c[:,:-1])*.5,weight_c[...,1:-1].detach(),n_f).sort()
    # points_f = ray_o.view(b,1,3) + t_f.view(b,n_f,1)*ray_d.view(b,1,3)
    # #import pdb;pdb.set_trace()
    points_f,_ = torch.cat([points_c,points_f],dim=-2).sort()
    t_f,_      = torch.cat([t_c,t_f],dim=-1).sort()
    n_f += n_c

    rgb_f,sigma_f = torch.split(get_rgbsigma(model_fine,points_f.view(-1,3),ray_d.view(b,1,3).expand([b,n_f,3]).reshape(-1,3)) , 3 , dim=-1)
    rgb_f,sigma_f = rgb_f.view(-1,n_f,3) , sigma_f.view(-1,n_f) # rgb:[b x n x 3], sigma[b x n]
    
    


    rgb_map_f,weight_f = integral_rgb(rgb_f,sigma_f,t_f)  #[bx3]
    
    
    if coarse_ratio is None: return rgb_map_c , rgb_map_f

    rgb_final = coarse_ratio * rgb_map_c + (1.0-coarse_ratio) * rgb_map_f
    assert 0<=rgb_final.min() and rgb_final.max() <= 1.0 , f"min={rgb_final.min()} , max={rgb_final.max()}"
    return rgb_final

def render_full_image( model_coarse:nn.Module,
                       model_fine:Optional[nn.Module],
                       h,w,K,pose,
                       n_samples_coarse=64 , 
                       n_samples_fine=0 , 
                       t_near=1e-8,
                       t_far=1.0,
                       coarse_ratio=0.01,
                       ray_batch_size=1<<11):
    
    rays_o,rays_d  = ray.get_rays(h,w,K,pose)
    rays_o, rays_d = rays_o.reshape(-1,3) , rays_d.reshape(-1,3)
    rgbs = []
    with torch.no_grad():     
        for i in tqdm.trange(0,rays_o.shape[0],ray_batch_size):
            rgb = render(model_coarse=model_coarse,
                         model_fine=model_fine,
                         ray_o=rays_o[i:i+ray_batch_size],
                         ray_d=rays_d[i:i+ray_batch_size],
                         n_samples_coarse=n_samples_coarse,
                         n_samples_fine=n_samples_fine,
                         t_near=t_near,
                         t_far=t_far,
                         coarse_ratio=coarse_ratio)
            rgbs.append(rgb)
        rgbs = torch.concat(rgbs,dim=0).view(h,w,3).permute([2,0,1])

    return rgbs


