

import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerMLP(nn.Module):

    def __init__(self, input_features:int , output_features:int , hidden_layers=200):
        
        super().__init__()
        i,o = input_features,output_features
        h = hidden_layers

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(i,h),
            nn.ReLU(),
            nn.Linear(h,o)
        )
    def forward(self,x:torch.Tensor):
        return self.layers(x)
