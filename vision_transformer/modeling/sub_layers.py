import torch, time
import torchvision
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from transformer.randomaug import RandAugment
from einops.layers.torch import Rearrange, Reduce
#/home/visionteam/tf_tutorials/attention-is-all-you-need-pytorch/train.py

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention with optional shape-bias penalty. '''

    def __init__(self, temperature, attn_dropout=0.1, alpha=1.0, dist_scale=1.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.alpha = alpha
        self.dist_scale = dist_scale

    def forward(
        self, 
        q, k, v,                 # q, k, v: (B, heads, N, d)
        mask=None,               # optional attention mask
        patch_positions=None,    # (B, N, 2) or (N, 2) if single-image
        patch_embeddings=None    # (B, N, d) or (N, d)
    ):
        """
        q, k, v : (B, H, N, D)
        patch_positions: for shape bias (B, N, 2) if you have a batch
        patch_embeddings: for shape bias (B, N, E) or (B, N, D)
        """

        # (B, H, N, d) x (B, H, d, N) -> (B, H, N, N)
        attn_logits = torch.matmul(q / self.temperature, k.transpose(2, 3))

        B, H, N, _ = q.shape

        # ---------------------------------------------------
        # (1) Optionally compute shape-bias penalty per batch
        # ---------------------------------------------------
        if patch_positions is not None and patch_embeddings is not None:
            # We'll assume patch_positions and patch_embeddings are batched
            # shapes: (B, N, 2), (B, N, D)
            # We'll produce a penalty matrix for each item in batch
            # penalty_matrices = []
            # for b_idx in range(B):
            #     # shape: (N,2) and (N,d)
            #     pos_b = patch_positions[b_idx]
            #     emb_b = patch_embeddings[b_idx]
            #     # compute penalty for this sample
            #     M = compute_shape_bias_penalty(
            #         pos_b, emb_b[1:,:], alpha=self.alpha, dist_scale=self.dist_scale
            #     )
            #     # M is (N, N)
            #     penalty_matrices.append(M.unsqueeze(0))  # -> (1, N, N)

            # # Stack into (B, N, N) and expand heads: (B, 1, N, N) -> (B, H, N, N)
            # penalty_full = torch.stack(penalty_matrices, dim=0)
            # #print(penalty_full.shape)  ##<---- 2x1x64x64
            # #penalty_full = penalty_full.unsqueeze(1).expand(-1, H, -1, -1)
            # penalty_full = penalty_full.expand(-1, H, -1, -1)
            
            # # Subtract from attn_logits
            # # attn_logits shape = (B, H, N+1, N+1)
            # # penalty_full shape = (B, H, N, N)
            # attn_logits[..., 1:, 1:] = attn_logits[..., 1:, 1:] - penalty_full
            # #print(attn_logits.shape, penalty_full.shape)
            # #attn_logits = attn_logits - penalty_full
            patch_embeddings_wo_cls = patch_embeddings[:, 1:, :] # (B, N, d)

            # Then call the new vectorized function
            penalty_batched = compute_shape_bias_penalty_batched(
                patch_positions,
                patch_embeddings_wo_cls,
                alpha=self.alpha,
                dist_scale=self.dist_scale
            )  # (B, N, N)
            # Suppose penalty.shape = (B, N, N)
            # You can expand to (B, H, N, N)
            penalty_expanded = penalty_batched.unsqueeze(1).expand(-1, H, -1, -1)  # (B, H, N, N)
            #print("penalty")
            #print(penalty_expanded.cpu().max(), penalty_expanded.cpu().min(), penalty_expanded.cpu().median())
            #print(attn_logits.cpu().max(), attn_logits.cpu().min(), attn_logits.cpu().median())
            attn_logits[..., 1:, 1:] -= penalty_expanded  # if you're skipping CLS in the attention
            #print(attn_logits.max(), attn_logits.median(), attn_logits.min())

        # -----------------------------------------------
        # (2) If you have a mask, apply it here
        # -----------------------------------------------
        if mask is not None:
            # mask shape is typically (B, 1, N, N) or (B, H, N, N), 
            # with 0/1 entries
            attn_logits = attn_logits.masked_fill(mask == 0, -1e9)

        # -----------------------------------------------
        # (3) Softmax over last dim
        # -----------------------------------------------
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        # -----------------------------------------------
        # (4) Multiply by v to get final output
        # -----------------------------------------------
        output = torch.matmul(attn, v)  # (B, H, N, D)

        return output  # or (output, attn) if you want the attention map



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, alpha=1.0, dist_scale=1.0):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, alpha=alpha, dist_scale=dist_scale)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None, patch_positions=None, patch_embeddings=None):
                                          # (B, N, 2) or (N, 2) if single-image

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv) = (1 x 65 x 512)
        # Separate different heads: b x lq x n x dv
        #print(q.shape)
        #print(self.w_qs.weight.shape)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) #(1 x 65 x 512) --> (1 x 65 x 8 x 64)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k) #(1 x 65 x 512) --> (1 x 65 x 8 x 64)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v) #(1 x 65 x 512) --> (1 x 65 x 8 x 64)

        # Transpose for attention dot product: b x n x lq x dv = (1 x 8 x 65 x 64)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        #q, attn = self.attention(q, k, v, mask=mask) # (1 x 8 x 65 x 64) and (1 x 8 x 65 x 65)
        q  = self.attention(q, k, v, mask=mask, patch_positions=patch_positions, patch_embeddings=patch_embeddings) # (10 x 8 x 12 x 64) and (10 x 8 x 12 x 12)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # q.transpose(1, 2) --> (1 x 65 x 8 x 64)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1) #(1 x 65 x 512)
        q = self.dropout(self.fc(q)) ## (1 x 65 x 512) x (512 x 512) --> (1 x 65 x 512)
        q += residual

        q = self.layer_norm(q)

        #return q, attn
        return q

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class ClassificationHeadWithAvgPooling(nn.Module):
    def __init__(self, d_model: int = 512, n_classes: int = 10):
        super().__init__()
        self.reduction_layer = Reduce('b n e -> b e', reduction='mean')
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear_layer = nn.Linear(d_model, n_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced_encoder_op = self.reduction_layer(x)
        #print('Reduced encoder shape:{}'.format(reduced_encoder_op.shape))
        layer_normed_reduced = self.layer_norm(reduced_encoder_op)
        output = self.linear_layer(layer_normed_reduced)
        return output


class ClassificationHeadWithLearnablePooling(nn.Module):
    def __init__(self, d_model: int = 512, n_classes: int = 10):
        super().__init__()
        self.reduction_layer = nn.Linear(d_model, 1)
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear_layer = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ### Learnable pooling
        reduction_layer_op = self.reduction_layer(x)
        reduction_layer_op = reduction_layer_op.transpose(-1, 1)
        reduction_layer_op = F.softmax(reduction_layer_op, dim=-1)
        enc_output = torch.matmul(reduction_layer_op, x)
        enc_output = enc_output.squeeze(1) #converts torch.Size([2, 1, 512]) -> torch.Size([2, 512])
        ## end of learnable pooling

        ## layer norm and a fully connected layer
        layer_normed_reduced = self.layer_norm(enc_output)
        output = self.linear_layer(layer_normed_reduced)
        #output = self.linear_layer(enc_output)
        return output


class ClassificationHeadClsTokenPooling(nn.Module):
    def __init__(self, d_model: int = 512, n_classes: int = 10):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear_layer = nn.Linear(d_model, n_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced_encoder_op = x[:,-1,:]
        #print('Reduced encoder shape:{}'.format(reduced_encoder_op.shape))
        layer_normed_reduced = self.layer_norm(reduced_encoder_op)
        output = self.linear_layer(layer_normed_reduced)
        return output


def make_classifier_head(classifier_type='average_pooling', d_model=512, n_classes=10):
    if(classifier_type == 'average_pooling'):
        classifier_head = ClassificationHeadWithAvgPooling(d_model=d_model, n_classes=n_classes)
    elif(classifier_type == 'learnable_pooling'):
        classifier_head = ClassificationHeadWithLearnablePooling(d_model=d_model, n_classes=n_classes)
    elif(classifier_type == 'cls_token_pooling'):
        classifier_head = ClassificationHeadClsTokenPooling(d_model=d_model, n_classes=n_classes)
    else:
        print('classifier_type of type:{} is not yet supported'.format(classifier_type))
    return classifier_head
