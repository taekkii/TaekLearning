
import torch
from torch.utils.data import Dataset
import numpy as np

from typing import Optional
from typing import Callable

from dataset.posedimg import PosedImage

class SkewedPosedImage(PosedImage):
    

    def pdf(self,x,sigma):
        return (2.*3.141592* sigma**2 )**-0.5 * np.exp(-0.5 * x * (sigma**-2.0))
    
    def __init__(self , path_dir:str , dataset_type:str = "blender", transform: Optional[Callable] = None , metadata_filename=None,load_device='folder', 
                 idx_max=10000, deviation = 1.0,pivot_axis = np.array([0.,1.,0.])):
        
        super().__init__(path_dir,dataset_type,transform,metadata_filename,load_device)
        prob = []
        for pose in self.poses:
            lookat = pose.cpu()@np.array([0.,0.,-1.,0.])
            cos_theta = lookat[:-1] @ pivot_axis
            prob.append(self.pdf(cos_theta,deviation))
        
        
        prob=np.array(prob)
        prob/=prob.sum()
        prob*=idx_max
        prob=prob.astype(int)
        print("Skewed probabilities:", prob)
        cum_prob = np.cumsum(prob).astype(int)

        self.idx_map = []
        j=0
        
        
        #NAIVE METHOD ALERT
        for i in range(idx_max):
            while j<len(self.poses) and cum_prob[j]<i:
                j+=1
            self.idx_map.append(min(j,len(self.poses)-1))

        self.idx_max = idx_max
        print("[SKEW DATASET] Ready")

    def __getitem__(self, idx):
        return super().__getitem__(self.idx_map[idx])
        
    def __len__(self):
        return self.idx_max