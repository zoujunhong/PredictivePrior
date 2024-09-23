from .networks import FFN
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

class FFN_bias(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim))
        
        self.norm = nn.LayerNorm(input_dim)
        self.modulate = nn.Linear(input_dim, 2 * input_dim)
    
    def forward(self, x, control):
        gain, bias = self.modulate(control).split([self.input_dim, self.input_dim], dim=-1)
        x = (1 + gain) * self.norm(x) + bias
        x = x + self.ffn(x)
        return x
    
def load_and_freeze(model: nn.Module, dict_name):
    dict = torch.load(dict_name, map_location='cpu')
    model.load_state_dict(dict, strict=True)
    stop_grad(model)

def stop_grad(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = False

class LinearNorm(nn.Module):
    def __init__(self, inplanes, planes, act=True, affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(planes, elementwise_affine=affine)
        self.conv = nn.Linear(inplanes, planes)
        self.act = act
    def forward(self, x):
        x = self.norm(self.conv(x))
        return F.gelu(x) if self.act else x

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

"""Slot Attention-based auto-encoder for object discovery."""
class DINOPredictor(nn.Module):
    def __init__(self, dim=384, res=28):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, res*res, dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1], res)
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        self.dim=dim
        self.res=res
        extension=4
        self.predictor = nn.ModuleList(
            [FFN_bias(dim, dim*extension),
            FFN_bias(dim, dim*extension),
            FFN_bias(dim, dim*extension),
            FFN_bias(dim, dim*extension),
            FFN_bias(dim, dim*extension),
            FFN_bias(dim, dim*extension)])


    def forward(self, x):
        b, n, c = x.shape
        x = x.permute(0,2,1).reshape(b,self.dim,self.res,self.res)
        pe = torch.repeat_interleave(self.pos_embedding, b, dim=0).permute(0,2,1).reshape(b, self.dim, self.res, self.res)

        source_grid = torch.rand([b, 8, 8, 2], device=x.device) * 2 - 1
        target_grid = torch.rand([b, 8, 8, 2], device=x.device) * 2 - 1
        
        source_feat = F.grid_sample(x, source_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1) # [b, 256, c]
        
        target_feat = F.grid_sample(x, target_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1) # [b, 256, c]
        target_pos  = F.grid_sample(pe, target_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1) # [b, 256, c]
        
        # W = F.cosine_similarity(source_feat, target_feat, dim=-1) # 
        # W[W>0.2] = 1
        # W[W<0.2] = 0
        # W = W * torch.mean(source_feat**2, dim=-1) * torch.mean(target_feat**2, dim=-1)
        # W[source_mask != target_mask] = 0.
        
        for i in range(len(self.predictor)):
            target_pos = self.predictor[i](target_pos, source_feat) # [b, 256, c]
        
        loss = torch.mean((1 - torch.cosine_similarity(target_pos, target_feat, dim=-1)))
        return loss
    
    
    def forward_test(self, x, source_grid, target_grid):
        b, n, c = x.shape
        x = x.permute(0,2,1).reshape(b,self.dim,self.res,self.res)   
        pe = torch.repeat_interleave(self.pos_embedding, b, dim=0).permute(0,2,1).reshape(b, self.dim, self.res, self.res)
        
        source_feat = F.grid_sample(x, source_grid, mode='bilinear', align_corners=True, padding_mode="border").flatten(2,3).permute(0,2,1) # [b, 256, c]
                
        target_feat = F.grid_sample(x, target_grid, mode='bilinear', align_corners=True, padding_mode="border").flatten(2,3).permute(0,2,1) # [b, 256, c]
        target_pos = F.grid_sample(pe, target_grid, mode='bilinear', align_corners=True, padding_mode="border").flatten(2,3).permute(0,2,1) # [b, 256, c]
        
        for i in range(len(self.predictor)):
            target_pos = self.predictor[i](target_pos, source_feat) # [b, 256, c]
        return torch.cosine_similarity(target_pos, target_feat, dim=-1), source_feat, target_feat
    
    def forward_whole(self, x):
        b, n, c = x.shape
        x = x.permute(0,2,1).reshape(b,c,28,28)
        pe = torch.repeat_interleave(self.pos_embedding, b, dim=0).permute(0,2,1).reshape(b, 384, 28, 28)
        
        source_grid = torch.rand([b, 20, 20, 2], device=x.device) * 2 - 1
        target_grid = torch.rand([b, 20, 20, 2], device=x.device) * 2 - 1
        
        source_feat = F.grid_sample(x, source_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1) # [b, 256, c]
        source_pos = F.grid_sample(pe, source_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1) # [b, 256, c]
        
        target_feat = F.grid_sample(x, target_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1) # [b, 256, c]
        target_pos = F.grid_sample(pe, target_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1) # [b, 256, c]
        
        for i in range(len(self.predictor)):
            target_pos = self.predictor[i](target_pos, source_feat)
        
        for i in range(len(self.predictor)):
            source_pos = self.predictor[i](source_pos, target_feat)
            
        pred_sim_1 = torch.cosine_similarity(target_pos, target_feat, dim=-1)
        
        pred_sim_2 = torch.cosine_similarity(source_pos, source_feat, dim=-1)
        
        pred_sim_1[pred_sim_1 > pred_sim_2] = pred_sim_2[pred_sim_1 > pred_sim_2]
        
        # dino_sim = torch.cosine_similarity(source_feat, target_feat, dim=-1)
        return pred_sim_1, source_grid, target_grid
    