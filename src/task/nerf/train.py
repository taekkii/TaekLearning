
import trainer
from . import ray
from . import render
from .utils import get_intrinsic

import torch

from typing import Optional , Callable

import utils.debug as dbg

def mean_squared_error(c_gt,c_render):
    return ((c_gt-c_render)**2).mean()
import tqdm

@torch.no_grad()
def psnr(mse):
    return -10. * torch.log(mse) / torch.log(10*torch.ones(1,device=mse.device) )
#override
def forward(self,gt_imgs:torch.Tensor, rays_o:torch.Tensor , rays_d:torch.Tensor, loss_func:Callable , n_samples:int=64 , t_near:float=1e-8 , t_far:float=1.0):
    """
    NeRF Trainer overriden forward
    [ARGS]
        gt_rgbs: ground_truth rgbs [bx3]
        rays_o : origin of rays    [bx3]
        rays_d : direction of rays [bx3]
    [RETURNS]
        output
        mse_loss
    """
    assert hasattr(self,'net') , "Configure your model by calling config_network() first."
    
    self.net.train()
    output = render.render(model=self.net, ray_origin=rays_o.flatten(0,-2) , ray_direction=rays_d.flatten(0,-2) , n_samples=n_samples , t_near=t_near , t_far=t_far )
    
    loss = loss_func(output,gt_imgs)
    
    return output,loss


def random_sample_rays(rays_o:torch.Tensor , rays_d:torch.Tensor,batch_size:int,gt_rgb:Optional[torch.Tensor] = None):
    
    idx = torch.randperm(rays_o.shape[0],device=rays_o.device)[:batch_size]
    if gt_rgb is not None: return rays_o[idx] , rays_d[idx] , gt_rgb[idx]
    else                 : return rays_o[idx] , rays_d[idx]

class NeRFTrainer(trainer.Trainer):

    
    #override
    def start(self):
        

        self.n_samples = self.settings['n_samples']
        self.t_near    = self.settings['t_near']
        
        self.ray_batch_size=(1<<19) // (self.n_samples)  # temporary: (2^19) can be larger with more powerful GPU

        if 't_far' not in self.settings or self.settings['t_far'] is None:
            self.t_far = self.get_t_far()
        else:
            self.t_far = self.settings['t_far']
        
        self.visualizer.attach_graph('Loss',self.record("loss") , "train")
        self.visualizer.attach_graph('PSNR',self.record('psnr') , "train")
        
        self.visualizer.attach_graph('Learning_Rate',self.record('lr'),'lr')
        self.count_rays = 0
        #dbg.on() 


        
    #override
    def step(self,data):
        
        dbg.stamp('data_load')
        train_chunk = self.trainchunks[0] # temporary

        gt_imgs,poses,fovx = data
        fovx = fovx[0]
        
        
        gt_imgs = gt_imgs.to(self.settings['device'])
        _,_,h,w = gt_imgs.shape

        dbg.stamp('get_shape')
        K = get_intrinsic(h,w,fovx)
        
        gt_rgbs = gt_imgs.permute([0,2,3,1]).reshape(-1,3)        
 
        rays_o, rays_d = ray.get_batch_rays(h,w,K,poses) #[b x h x w x 3]
        rays_o, rays_d = rays_o.to(self.settings['device']), rays_d.to(self.settings['device'])
        rays_o, rays_d = rays_o.reshape(-1,3) , rays_d.reshape(-1,3)
        rays_o, rays_d, gt_rgbs = random_sample_rays(rays_o , rays_d, self.ray_batch_size,gt_rgb=gt_rgbs) 
        dbg.stamp("data_prepare")

        
        
        _,loss = forward(train_chunk,gt_rgbs, rays_o , rays_d , 
                         loss_func=mean_squared_error,
                         n_samples=self.n_samples ,
                         t_near=self.t_near , t_far=self.t_far )
        #dbg.stamp('forward')
        
        train_chunk.backward(loss)
        dbg.stamp('backward')

        self.record("loss",loss.detach())
        self.record("psnr",psnr(loss).detach())
        self.record('lr',train_chunk.get_lr())
        
        dbg.stamp('wait')
        dbg.wait(10)

        
        #tmp_validate
        if self.it % 500 == 0:
            v = self.visualizer.visdom
            print("[Validate] it={}".format(self.it))
           
            rays_o,rays_d = ray.get_rays(h,w,K,poses[0])
            rays_o, rays_d = rays_o.to(self.settings['device']), rays_d.to(self.settings['device'])
            rays_o, rays_d = rays_o.reshape(-1,3) , rays_d.reshape(-1,3)
            rgbs = []
            with torch.no_grad():     
                for i in tqdm.trange(0,rays_o.shape[0],self.ray_batch_size):
                    rgb = render.render(train_chunk.net,
                                        rays_o[i:i+self.ray_batch_size],
                                        rays_d[i:i+self.ray_batch_size],
                                        n_samples=self.n_samples,
                                        t_near=self.t_near,
                                        t_far =self.t_far)
                    rgbs.append(rgb)
                rgbs = torch.concat(rgbs,dim=0).view(h,w,3).permute([2,0,1])
            
            v.images(torch.stack([rgbs,gt_imgs[0]]),win="validate",env=self.settings['experiment_name'])
            print()
        
    
    def validate(self):
        pass
    
    def get_t_far(self):
        return 1.0 # temporary
