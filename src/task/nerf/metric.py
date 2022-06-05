

import torch
import torch.nn as nn


import tqdm
from torch.utils.data import DataLoader
from typing import Optional
from typing import Callable


@torch.no_grad()
def psnr( img, pose, net:nn.Module):
    pass



@torch.no_grad()
def psnr_eval(dataloader:DataLoader , net:nn.Module , objective_function:Optional[Callable] = None):

    net.eval()
    for i,(img,pose) in tqdm(enumerate(dataloader)):
        pass