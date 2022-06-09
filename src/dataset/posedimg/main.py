

from torch.utils.data import Dataset
import numpy as np

import json
import os
import imageio

import torch
import torchvision.transforms
import torchvision.datasets.folder

from typing import Optional,Callable


DEFAULT_BLENDER_JSON_FILENAME = "transforms"
 

class UnsupportedDatasetTypeError(Exception):
    pass

def load_blender(path_dir:str , json_filename:str):
    """
    Load blender file configured by [json_filename].json
    
    [ARGS]
      path_dir: root directory
      json_filename: json file. Both of having ".json" at the end of filename or not are okay.
    
    [RETURNS]
      - Paths of images. Note that these are not: RGB numpy arrays[h x w x 3]. 
      - poses [B x 4 x 4] np array 
      - fov_angle [float]
    """


    if len(json_filename)<5  or  json_filename[-5:] != '.json':
        json_filename += '.json'

    
    path_json = os.path.join(path_dir , json_filename)
    
    with open(path_json,'r') as fp:
        meta_data = json.load(fp)
    
    
    image_paths = []
    poses = []

    for frame in meta_data['frames']:

        image_path = os.path.join(path_dir,frame['file_path']+'.png')
        image_paths.append( image_path )

        poses.append(np.array(frame['transform_matrix']))
    
    
    poses = np.array(poses).astype(np.float32)
    fov_angle =float(meta_data['camera_angle_x'])

    return image_paths , poses , fov_angle



class PosedImage(Dataset):
    
    def __init__(self , path_dir:str , dataset_type:str = "blender", transform: Optional[Callable] = None , metadata_filename=None,load_device='folder'):

        
        super().__init__()
        assert load_device=='folder'  or  'cuda' in load_device
        self.load_device = load_device

        self.transform = transform
        print(f"Loading PosedImage, type[{dataset_type}]")
        

        # [Blender loader]
        if dataset_type == "blender":         
            if metadata_filename is None:
                metadata_filename = DEFAULT_BLENDER_JSON_FILENAME
            
            self.image_paths , self.poses , self.fov_angle = load_blender(path_dir, metadata_filename)
            

            
        else:
            raise UnsupportedDatasetTypeError("Not supported dataset")
        
        if 'cuda' in load_device:
            print("===== Batching all images to {} =====".format(load_device))
            print("[WARNING] If size of images are large, loading dataset to gpu is heavily depreciated!!\n")
            
            imgs = []
            for image_path in self.image_paths:
                img = torchvision.datasets.folder.default_loader(path=image_path)
                img = torchvision.transforms.ToTensor()(img)
                imgs.append(img)
            
            self.imgs = torch.stack(imgs).to(load_device)
            self.poses = torch.from_numpy(self.poses).to(load_device)
            self.imgs.requires_grad=False
            self.poses.requires_grad=False
        
    def __getitem__(self,idx):
        if self.load_device=='folder':
            img = torchvision.datasets.folder.default_loader(path=self.image_paths[idx]) # loads from path
            pose = self.poses[idx]
        else: 
            img,pose = self.imgs[idx] , self.poses[idx]
        
        if self.transform:
            img = self.transform(img)
        return img, pose, self.fov_angle

    def __len__(self):
        return len(self.poses)
