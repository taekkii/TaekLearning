
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torchvision.transforms


from dataset.posedimg import PosedImage
from visdom import Visdom

from task.nerf.ray import get_batch_rays, get_rays
from task.nerf.utils import get_intrinsic


class MyDataset(Dataset):
    def __init__(self,device):
        super().__init__()

        self.device=device
        self.x = torch.randn(50000,10000 , device=self.device)
        self.x.requires_grad=False

    def __getitem__(self, index):
        return self.x[index]
    
    def __len__(self):
        return 50000

import time
dset = MyDataset('cuda:5')
loader = DataLoader(dset,batch_size=64,shuffle=True,num_workers=0)
t0 = time.time()
for batch_data in loader:
    y=2*batch_data
    

print(time.time()-t0)

# lego_dset = PosedImage("/home/jeongtaekoh/dataset/lego_downscale/lego200" , transforms=torchvision.transforms.ToTensor() ,metadata_filename="transforms_train")

# loader = DataLoader(lego_dset , batch_size=2,shuffle=True)


# v = Visdom()
# for i, (img,pose,fovx) in enumerate(loader):
#     print("[ITERATION] {}".format(i))
#     fovx = fovx[0]
#     b,ch,h,w = img.shape
    
    
#     v.images(img)
#     if i >= 0: break



# class MyDataset(Dataset):
#     def __init__(self):
#         print("hi")
    
#     def __getitem__(self,idx):
#         return np.array([idx])
#     def __len__(self):
#         return int(1e12)


# dset = MyDataset()
# loader = DataLoader(dset,batch_size=4,shuffle=True)

# for i,data in enumerate(loader):
#     print(data)
#     if i>=100:
#         break












# #import cvxpy as cp
# from time import sleep
# import numpy as np


# import matplotlib.pyplot as plt


# n = 10

# # Variables

# w = np.array([0.0 , 0.0])
# b = 0.0

# # w = cp.Variable(2)
# # b = cp.Variable(1)
# # xi = cp.Variable(n)



# # DATASET
# x = np.array( [ [0,0.5] ,
#                 [2,4],
#                 [4,3],
#                 [5,3.5],
#                 [2,1],
#                 [3,4.5],
#                 [-1,-1],
#                 [-2,0],
#                 [-3,1],
#                 [1,3] 
#                ])
# y = np.array( [-1,-1,-1,-1,-1,1,1,1,1,1])

# xx=np.linspace(-5,5)


# def get_grad_w(x,y,w,b,C):
#     u = np.array(w)
#     # could be more fancy numpy-like implementation, but use naive method for this HW
#     for i in range(x.shape[0]):

#         # consider y[i] * (w@x[i]+b) < 1  : okay by itself
#         # consider y[i] * (w@x[i]+b) == 1 : gradient NOT exist, but any subgradient can be used, so it's okay

#         if y[i] * (w@x[i]+b) <= 1.0: u -= C * y[i]*x[i]
    
#     return u

# def get_grad_b(x,y,w,b,C):
#     cnt = 0
#     # could be more fancy numpy-like implementation, but use naive method for this HW
#     for i in range(x.shape[0]):
#         if y[i] * (w@x[i]+b) <= 1.0: cnt+=y[i]
#     return C * -cnt

# def f0(x,y,w,b,C):
#     return w@w + C * np.sum( np.array( [ max( 1-y[i]*(w@x[i]+b) , 0 ) for i in range(x.shape[0]) ] ) )  
  

# def get_step_size(x,y,w,b,C,grad_w,grad_b , alpha=0.3 , beta = 0.995):
    
#     # backtracking line search!
  
#     t = 1.0
    
#     while f0(x,y, w - t*grad_w , b - t*grad_b , C) > f0(x,y,w,b,C) - alpha * t * (grad_w@grad_w + grad_b*grad_b) :
#         t *= beta
#         # print(f"lh:{f0(x,y, w - t*grad_w , b - t*grad_b , C):.6f}")
#         # print(f"rh:{f0(x,y,w,b,C) - alpha * t * (grad_w@grad_w + grad_b*grad_b):.6f}")
#         # print("t=",t)
#         # print()
#         # sleep(0.5)
#     return t
#     #  return 0.001 # comeback later
# def solve(x,y,w,b,C , num_iters = 10):
    
#     print(f"iter: {0:5d} , loss: {f0(x,y,w,b,C):5.2f}")
#     for iter in range(1,num_iters+1):
        
#         # print(w)
#         # print(b)
#         # print("="*30)
#         grad_w = get_grad_w(x,y,w,b,C)
#         grad_b = get_grad_b(x,y,w,b,C)
#         t = get_step_size(x,y,w,b,C,grad_w,grad_b)
        
#         w -= t*grad_w
#         b -= t*grad_b
        
#         if iter%1 == 0:
#             print(f"iter: {iter:5d} , loss: {f0(x,y,w,b,C):7.5f} , w = {w} , b = {b} , t = {t:7.5f}")
#     return w,b


# #for C in [0.2 , 0.4 , 1 , 2]:
# for C in [1]:

#     # objective = cp.Minimize(0.5 * cp.norm(w,2)**2   +   C * cp.sum(xi) )
#     # constraints = [ y[i]*(w.T@x[i] - b) >= 1 - xi[i] for i in range(n) ] +   \
#     #               [ xi[i] >= 0 for i in range(n) ]

#     # prob = cp.Problem(objective=objective , constraints=constraints)
#     # result = prob.solve()

#     w,b = solve(x,y,w,b,C)

#     #REPORT
#     print("C =",C)
#     print("w =",w)
#     print("b =",b)
# #   print("xi=",xi.value)
#     print("="*20+'\n')
    
#     yy = (-w[0]*xx + b) / w[1]
#     yy_plus1 = (-w[0]*xx + b+1) / w[1]
#     yy_minus1 = (-w[0]*xx + b-1) / w[1]
    
#     #PLOT

#     plt.title(f"C={C}")
#     plt.plot(xx,yy,'-',xx,yy_plus1,'--',xx,yy_minus1,'--')
#     plt.scatter(x[:,0],x[:,1],c=y)
#     plt.show()  
    
