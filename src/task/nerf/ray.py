
import torch
import numpy as np




# Ray helpers
# [CREDIT] credit by yenchenlin
def get_rays(H, W, K, c2w):
    device=c2w.device
    i, j = torch.meshgrid(torch.linspace(0, W-1, W,device=device), torch.linspace(0, H-1, H,device=device),indexing='ij')  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i,device=device)], -1) #dirs[b x h x w x 3]


    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[ :3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

# [CREDIT] yenchenlin, modified by taekkii
def get_batch_rays(H, W, K, c2w:torch.Tensor):
    """
    H,W:INT
    K: [3x3] torch
    c2w:[b x 4 x 4 ] pose matrices
    """
    device = c2w.device

    i, j = torch.meshgrid(torch.linspace(0, W-1, W,device=device), torch.linspace(0, H-1, H,device=device),indexing='ij')  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i,device=device)], -1) #dirs[b x h x w x 3]
           
    
    rays_d = torch.sum( dirs.view(-1,H,W,1,3) 
                      *(c2w.view(-1,1,1,4,4))[ : , : , : ,:3,:3], dim=-1)  #[b x h x w x 1 x 3 ] * [ b x 1 x 1 x 3 x 3 ] = [ b x h x w x 3 x 3 ] --(sum)--> [b x h x w x 3]
    rays_o = (c2w[:,:3,-1]).view(-1,1,1,3).expand(rays_d.shape)
    return rays_o, rays_d


# [CREDIT] credit by yenchenlin
def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d