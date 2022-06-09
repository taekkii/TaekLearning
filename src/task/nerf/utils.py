

import numpy as np
import torch

def get_focal(width:float , fovx:float):
    """
        get focal(f) from W and fovx(so called "camera_angle_x")
        [ARGS]
            width: number of horizontal pixels
            fovx(camera_angle_x): fovx angle
        
        [RETURNS]
            corresponding focal length
    """
    return .5 * width/np.tan(.5*fovx)

def get_intrinsic(h:float, w:float, fovx:float , numpy=False):
    """
        get [3x3] intrinsic matrix(K) from h, w and fovx(so called "camera_angle_x")
        [ARGS]
            height:number of vertical pixels
            width: number of horizontal pixels
            fovx(camera_angle_x): fovx angle
            numpy: gives [3x3] numpy array instead of pytorch tensor when equals True.

        [RETURNS]
            corresponding intrinsic matrix(K)
    """
    f = get_focal(w,fovx)
    
    
    if numpy:
        return np.array([[f  , 0. , .5*w],
                         [0. , f  , .5*h],
                         [0. , 0. ,  1. ]])
    
    return torch.Tensor([[f  , 0. , .5*w],
                         [0. , f  , .5*h],
                         [0. , 0. ,  1. ]])
