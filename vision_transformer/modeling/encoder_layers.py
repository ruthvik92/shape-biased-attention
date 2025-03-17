import math
import torch
import numpy as np
import torchvision
import time
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from transformer.randomaug import RandAugment
from torchsummary import summary
from einops import rearrange, reduce, repeat
from transformer.randomaug import RandAugment
from einops.layers.torch import Rearrange, Reduce

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1,alpha=1.0,):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout,alpha=alpha)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None, patch_positions=None, patch_embeddings=None):
        #enc_output, enc_slf_attn = self.slf_attn(
        #    enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask, patch_positions=patch_positions, patch_embeddings=patch_embeddings)
        if(type(enc_output) == tuple):
            enc_output, enc_slf_attn = enc_output
        enc_output = self.pos_ffn(enc_output)
        #return enc_output, enc_slf_attn
        return enc_output

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_conv_layers=1,alpha=1.0,):

        super().__init__()

        self.position_enc = PatchEmbedding(d_model=d_model, n_conv_layers=n_conv_layers,patch_size=16, img_size=224)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout,alpha=alpha) \
                                          for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model
        self.positions_2d = get_patch_positions(img_size=224, patch_size=16, device=device)

    def forward(self, input_image, return_attns=False):
        B = input_image.shape[0]

        enc_slf_attn_list = []

        # -- Forward
        patch_embedding = self.position_enc(input_image)
        #print(patch_embedding.shape)
        enc_output = self.dropout(patch_embedding)
        enc_output = self.layer_norm(enc_output)
        positions_2d_batched = self.positions_2d.unsqueeze(0).repeat(B, 1, 1)  # (B, 64, 2)

        for enc_layer in self.layer_stack:
            #enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=None)
            #enc_slf_attn_list += [enc_slf_attn] if return_attns else []
            enc_output = enc_layer(enc_output, slf_attn_mask=None, patch_positions=positions_2d_batched, patch_embeddings=patch_embedding)

        #if return_attns:
        #    return enc_output, enc_slf_attn_list
        return enc_output
