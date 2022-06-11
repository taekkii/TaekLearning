
import trainer
from . import ray
from . import render
from .utils import get_intrinsic

import torch

from typing import Optional , Callable

import utils.debug as dbg

def squared_error(c_gt,c_render):
    return ((c_gt-c_render)**2).mean()

@torch.no_grad()
def psnr(mse):
    return -10. * torch.log(mse) / torch.log(10.*torch.ones(1,device=mse.device) )

def random_sample_rays(rays_o:torch.Tensor , rays_d:torch.Tensor,batch_size:int,gt_rgb:Optional[torch.Tensor] = None):
    
    idx = torch.randperm(rays_o.shape[0],device=rays_o.device)[:batch_size]
    if gt_rgb is not None: return rays_o[idx] , rays_d[idx] , gt_rgb[idx]
    else                 : return rays_o[idx] , rays_d[idx]

class NeRFTrainer(trainer.Trainer):

    
    #override
    def start(self):
        

        self.n_samples = self.settings['n_samples']
        self.n_samples_fine = self.settings['n_samples_fine']
        self.t_near    = self.settings['t_near']
        self.coarse_ratio = self.settings['coarse_ratio']


        if 'ray_batch_size' in self.settings and self.settings['ray_batch_size'] is not None:
            self.ray_batch_size = self.settings['ray_batch_size']
        else:    
            self.ray_batch_size=(1<<19) // (self.n_samples + self.n_samples_fine)  # temporary: (2^19) can be larger with more powerful GPU
        
        if 't_far' not in self.settings or self.settings['t_far'] is None:
            self.t_far = self.get_t_far()
        else:
            self.t_far = self.settings['t_far']
        
        self.visualizer.attach_graph('Loss',self.record("loss") , "train")
        self.visualizer.attach_graph('PSNR',self.record('psnr') , "train")
        
        self.visualizer.attach_graph('Learning_Rate',self.record('lr'),'lr')
        
        self.trainchunk_c = self.trainchunks[0]
        self.trainchunk_f = None if len(self.trainchunks)<=1 else self.trainchunks[1]
        self.loss_sum=0

        
    #override
    def step(self,data):
        
        dbg.stamp('data_load')
        

        gt_imgs,poses,fovx = data
        fovx = fovx[0]
        
        
        gt_imgs = gt_imgs.to(self.settings['device'])
        _,_,h,w = gt_imgs.shape

        
        dbg.stamp('get_shape')
        K = get_intrinsic(h,w,fovx)
          
        
        rays_o, rays_d = ray.get_batch_rays(h,w,K,poses) #[b x h x w x 3]
        rays_o, rays_d = rays_o.to(self.settings['device']), rays_d.to(self.settings['device'])
        
        
        if self.it <= self.settings.get('center_crop_iteration',0):
            rays_o,rays_d = rays_o[:,h//4:h*3//4,w//4:w*3//4,:] , rays_d[:,h//4:h*3//4,w//4:w*3//4,:]
            gt_rgbs = gt_imgs[:,:, h//4:h*3//4 , w//4:w*3//4].permute([0,2,3,1]).reshape(-1,3)
        else:
            gt_rgbs = gt_imgs.permute([0,2,3,1]).reshape(-1,3)
        rays_o, rays_d = rays_o.reshape(-1,3) , rays_d.reshape(-1,3)
        rays_o, rays_d, gt_rgbs = random_sample_rays(rays_o , rays_d, self.ray_batch_size,gt_rgb=gt_rgbs) 
        dbg.stamp("data_prepare")

        rgbs_c,rgbs_f = render.render(model_coarse = self.trainchunk_c.net,
                                      model_fine = self.trainchunk_f.net,
                                      ray_o = rays_o,
                                      ray_d = rays_d,
                                      n_samples_coarse = self.n_samples,
                                      n_samples_fine   = self.n_samples_fine,
                                      t_near = self.t_near,
                                      t_far  = self.t_far)
        if rgbs_f is None: loss = squared_error(gt_rgbs , rgbs_c) 
        else             : loss = squared_error(gt_rgbs , rgbs_c) + squared_error(gt_rgbs,rgbs_f)
    
        loss.backward()
        for tc in self.trainchunks:tc.step()
        
        dbg.stamp('backward')

        mse = loss.detach()
        self.record("loss",mse)
        self.record("psnr",psnr(mse).detach())
        self.record('lr',self.trainchunk_c.get_lr())
       
        dbg.stamp('record')
        dbg.wait(10)

        
        #temporary
        self.loss_sum+=loss.item()
        if self.it%50 == 0:
            import tqdm
            print(f"[TRAIN] Iter: {self.it} Loss: {self.loss_sum/50}  PSNR: {psnr(torch.tensor(self.loss_sum/50)).item()}")
            self.loss_sum=0
        #tmp_validate
        if self.it % self.settings['i_validation'] == 0:
            v = self.visualizer.visdom
            print("[Validate] it={}".format(self.it))
           
            
            rgbs = render.render_full_image(self.trainchunk_c.net,
                                            self.trainchunk_f.net,
                                            h,w,K,poses[0],
                                            n_samples_coarse = self.n_samples,
                                            n_samples_fine   = self.n_samples_fine,
                                            t_near = self.t_near,
                                            t_far  = self.t_far)
            v.images(torch.stack([rgbs,gt_imgs[0]]),win=f"validate:{self.it}",env=self.settings['experiment_name'])
            print()
        
    
    def validate(self):
        pass
    
    def get_t_far(self):
        return 1.0 # temporary
