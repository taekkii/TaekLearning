

from torch.utils.data import Dataset
import numpy as np

import json
import os
import imageio


class UnsupportedDatasetTypeError(Exception):
    pass
class PosedImage(Dataset):
    
    def __init__(self,path_dir:str,dataset_type:str="blender"):
        print(f"[TaekLearning] Loading PosedImage, type = {dataset_type}")
        
        imgs = []
        poses = []

        # [Blender loader]
        # Credit: nerf-pytorch(yenchenlin)
        if dataset_type == "blender":
            
            with open(path_dir,'r') as fp:
                meta_data = json.load(fp)
            
            for frame in meta_data['frames']:
                image_path = os.path.join(path_dir,frame['file_path']+'.png')
                imgs.append(imageio.imread(image_path))
                poses.append(np.array(frame['transform_matrix']))
            self.imgs = (np.array(imgs) / 255.0).astype(np.float32)
            self.poses = np.array(poses).astype(np.float32)
            self.fov_angle =float(meta_data['camera_angle_x'])
        
        else:
            raise UnsupportedDatasetTypeError("Not supported dataset")
        
        
    def __getitem(self,idx):
        return self.imgs[idx], self.poses[idx]

    def __len__(self):
        return len(self.imgs)
