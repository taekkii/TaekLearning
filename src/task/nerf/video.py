
# [credit]yenchenlin nerf-pytorch
import torch
import torch.nn as nn
import numpy as np


import imageio
import os
from typing import Optional
from typing import Union
from . import render
from . import yenchenlinutils

import tqdm
import utils.system

def record(imgs,filename='record',dir='./video'):
    if len(filename)<4 or filename[-4:]!='.mp4':
        filename += '.mp4'
    utils.system.prepare_dir(dir)
    path = os.path.join(dir, filename)
    imageio.mimwrite(path , yenchenlinutils.to8b(imgs), fps=30, quality=8)
    
    print(f"\n[RECORD] finished. Path: {path}\n")

def spherical_record( record_name:str,
                      model_coarse:nn.Module,
                      model_fine:Optional[nn.Module],
                      h,w,K,
                      frames=40,
                      n_samples_coarse=64 , 
                      n_samples_fine=0 , 
                      t_near=1e-8,
                      t_far=1.0,
                      coarse_ratio=0.01,
                      ray_batch_size=1<<11):
   
    for param in model_coarse.parameters():
        device=param.device
        break
    poses = torch.stack([yenchenlinutils.pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,frames+1)[:-1]], 0).to(device)
    imgs = []
    for i,pose in enumerate(poses):
        print(F"[SPHERICAL_RECORDING] {i+1}/{poses.shape[0]}...")
        img = render.render_full_image( model_coarse,
                                        model_fine,
                                        h,w,K,pose,
                                        n_samples_coarse=n_samples_coarse,
                                        n_samples_fine=n_samples_fine,
                                        t_near=t_near,
                                        t_far=t_far,
                                        coarse_ratio=coarse_ratio,
                                        ray_batch_size=ray_batch_size  )

        imgs.append( img.cpu().permute([1,2,0]).numpy() )
        
    imgs=np.stack(imgs,axis=0)
    print(imgs.shape)
    record(imgs,'record_'+record_name)
