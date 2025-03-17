import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.randomaug import RandAugment
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat
from transformer.Optim import ScheduledOptim
from transformer.randomaug import RandAugment
from einops.layers.torch import Rearrange, Reduce

class ViT(nn.Module):
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, n_classes, 
                dropout=0.1, n_conv_layers=1,alpha=1.0,):
        super().__init__()
        self.encoder = Encoder(n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, 
                               d_inner=d_inner, n_conv_layers=n_conv_layers,alpha=alpha)
        self.classifier_head = ClassificationHeadWithAvgPooling(d_model=d_model, n_classes=n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(x.shape)
        #encoder_op = self.encoder(input_image=x)
        encoder_op = self.encoder(x)
        classifier_op = self.classifier_head(encoder_op)
        return classifier_op
