
import trainer
from . import ray
from . import render
from .utils import get_intrinsic

import torch

from typing import Optional , Callable

import utils.debug as dbg

def squared_error(c_gt,c_render):
    return ((c_gt-c_render)**2).mean()


def psnr(mse):
    return -10. * torch.log(mse) / torch.log(10.*torch.ones(1,device=mse.device) )

def balanced_mse(y,y_pred,sigma_noise):
    
    
    y_pred = y_pred.view(-1) #[3b]
    y = y.view(-1)           #[3b]

    tau = 2. * (sigma_noise**2.)


    numer = torch.exp( -( (y_pred - y)**2 ) / tau ) # [3b]

    denom = torch.exp( -(y_pred.view(-1,1)-y)**2  / tau ).sum(dim=-1) 
    loss_vec = -torch.log(numer/denom + 1e-6)
    
    
    return loss_vec.mean()


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

        self.net_c = self.trainchunk_c.net
        self.net_f = None if self.trainchunk_f is None else self.trainchunk_f.net
        
        
        self.add_on(self.summary,cycle=self.settings['i_summary'])
        
        # Optimization Theory Experiment #
        self.flag = {}
        if 'flag' in self.settings and type(self.settings['flag'])==str:
            import utils.dictionarylike as dictionarylike
            self.flag = dictionarylike.parse(self.settings['flag'])
        if 'balance_mse' in self.flag:
            if self.flag['balance_mse'] == 'train':
                print("[BALANCED MSE] with trainable parameter")
                self.sigma_noise = torch.tensor([0.1],device=self.settings['device'],requires_grad=True)
                self.trainchunk_c.optimizer.add_param_group({"params":self.sigma_noise})
            else:
                print("[BALANCED MSE] with sigma_noise={:.3f}".format(self.flag['balance_mse']))
                self.sigma_noise = torch.tensor([self.flag['balance_mse']],device=self.settings['device'])
        if self.flag.get('lambda',0) == 'train':
            print('[WEIGHTED LOSS:TRAINBLE]')
            self.lam = torch.tensor([1.0],device=self.settings['device'],requires_grad=True)
            self.trainchunk_c.optimizer.add_param_group({"params":self.lam})
    
    #override
    def step(self,data):
        
        dbg.stamp('data_load')
        
        self.net_c.train()
        if self.net_f: self.net_f.train()

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

        if 'center' in self.flag:
            t_center = (self.t_near+self.t_far)/2
            beta = self.it / self.settings['iter']
            t_near = beta*self.t_near + (1. - beta)*t_center
            t_far  = beta*self.t_far  + (1. - beta)*t_center
        else:
            t_near,t_far = self.t_near,self.t_far
        rgbs_c,rgbs_f = render.render(model_coarse = self.net_c,
                                      model_fine = self.net_f,
                                      ray_o = rays_o,
                                      ray_d = rays_d,
                                      n_samples_coarse = self.n_samples,
                                      n_samples_fine   = self.n_samples_fine,
                                      t_near = t_near,
                                      t_far  = t_far)
        if rgbs_f is None:
            loss = squared_error(gt_rgbs , rgbs_c) 
            psnr_it = psnr(loss).item()
        else:
            if 'lambda' in self.flag: lam = self.lam if self.flag['lambda']=='train' else self.flag['lambda']
            else: lam = 1.0
            
            if 'norm' in self.flag: norm=self.flag['norm']
            else: norm = 2.0

            def loss_fn(x,y,p):
                numel = x.numel()
                return ((x-y).norm(p=p)**2) / numel

            if 'iter' in self.flag:
                ratio = self.it/self.settings['iter']
                loss = (1. - ratio) * loss_fn(gt_rgbs , rgbs_c,norm) + ratio * lam * loss_fn(gt_rgbs,rgbs_f,norm)
            elif 'balance_mse' in self.flag:
            

                loss = balanced_mse(gt_rgbs,rgbs_c,self.sigma_noise) + lam * balanced_mse(gt_rgbs,rgbs_f,self.sigma_noise)
            else:
                loss = loss_fn(gt_rgbs , rgbs_c,norm) + lam * loss_fn(gt_rgbs,rgbs_f,norm)


            #loss = squared_error(gt_rgbs , rgbs_c) + squared_error(gt_rgbs,rgbs_f)
            psnr_it = psnr(squared_error(gt_rgbs,rgbs_f.detach())).item() if 'iter' not in self.flag            \
                else  psnr(squared_error(gt_rgbs,(1. - ratio)*rgbs_c.detach() + ratio * rgbs_f.detach() ) ).item()
       
        loss.backward()
    
        # if self.it%1==0:
            
        #     for name,param in self.net_c.named_parameters():
        #         if param.grad.norm()>0.0:
        #             print("[{}]".format(name))
        #             print(param.grad.norm().item())
        for tc in self.trainchunks:tc.step()
        
        dbg.stamp('backward')

        self.record("loss",loss.item())
        self.record("psnr",psnr_it)
        self.record('lr',self.trainchunk_c.get_lr())
       
        dbg.stamp('record')
        dbg.wait(10)
        
     
        #tmp_validate
        if self.it % self.settings['i_validation'] == 0:
            v = self.visualizer.visdom
            print("\n[Validate] it={}".format(self.it))
            
            ratio = 0.01
            if 'iter' in self.flag: ratio =  self.it/self.settings['iter']
            rgbs = render.render_full_image(self.net_c,
                                            self.net_f,
                                            h,w,K,poses[0],
                                            n_samples_coarse = self.n_samples,
                                            n_samples_fine   = self.n_samples_fine,
                                            t_near = t_near,
                                            t_far  = t_far,
                                            coarse_ratio=1.-ratio)
            v.images(torch.stack([rgbs,gt_imgs[0]]),win=f"validate",env=self.settings['experiment_name'])
            print(f"[VALIDATE] PSNR: {psnr(squared_error(gt_imgs[0],rgbs) ).item():8.5f}")
            print()
           # import pdb;pdb.set_trace()
        
    def summary(self):
        print(f"\n[TRAIN] Iter: {self.it} Loss: {sum(self.history['loss'][-50:])/50:8.5f}  PSNR: {sum(self.history['psnr'][-50:])/50:8.5f}")
        if 'balance_mse' in self.flag:
            print(f"[TRAIN] Sigma: {self.sigma_noise.item():8.5f}")
        if self.flag.get('lambda',0)=='train':
            print(f"[TRAIN] Lambda: {self.lam.item():8.5f}")

    def validate(self):
        pass
    
    def get_t_far(self):
        return 1.0 # temporary
