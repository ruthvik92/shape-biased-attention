import math
import torch
import numpy as np
import torchvision
import time
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int=3, patch_size: int=16, d_model: int=512, img_size: int=size,
                n_conv_layers: int=1):
        self.patch_size = patch_size
        super().__init__()
        # using a conv layer instead of a linear one -> performance gains
        # same_conv_layer means the shapes of input and output images is same as opposed to valid mode conv.
        self.same_conv_layer_stack = nn.ModuleList([nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2) \
                                                    for i in range(n_conv_layers)])
        self.conv_proj_layer = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.re_arrange_layer = Rearrange('b e (h) (w) -> b (h w) e')
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        #self.position_token = nn.Parameter(torch.randn((img_size // patch_size)**2 + 1, d_model))
        
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, *_ = x.shape
        for same_conv_layer in self.same_conv_layer_stack:
            x = same_conv_layer(x)
        convd_img = self.conv_proj_layer(x)
        #print('Output of convolution:{}'.format(convd_img.shape))
        re_arranged_ip = self.re_arrange_layer(convd_img)
        #print('Rearranged ip:{}'.format(re_arranged_ip.shape))
        cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)
        #print('CLS token:{}'.format(cls_token.shape))
        concated_ip = torch.cat([cls_token, re_arranged_ip], axis=1)
        #concated_ip += self.position_token
        #print('Concated ip:{}'.format(concated_ip.shape))
        
        return concated_ip
