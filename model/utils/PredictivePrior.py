from .networks import get_2d_sincos_pos_embed
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

class LinearNorm(nn.Module):
    def __init__(self, inplanes, planes, act=True, affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(planes, elementwise_affine=affine)
        self.conv = nn.Linear(inplanes, planes)
        self.act = act
    def forward(self, x):
        x = self.norm(self.conv(x))
        return F.gelu(x) if self.act else x

"""Slot Attention-based auto-encoder for object discovery."""
class DINOPredictor(nn.Module):
    def __init__(self, dim=384, res=16):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.dim = dim
        self.res = res
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.res*self.res, self.dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1], self.res)
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.norm = nn.LayerNorm(self.dim, elementwise_affine=False)
        
        self.predictor = nn.Sequential(
            LinearNorm(dim, 768),
            LinearNorm(768, 768),
            LinearNorm(768, 768),
            LinearNorm(768, 768),
            LinearNorm(768, 768),
            LinearNorm(768, dim))


    def forward(self, x):
        with torch.no_grad():
            b, n, c = x.shape
            x = self.norm(x)
            x = x.permute(0,2,1).reshape(b,c,self.res,self.res)
        pe = torch.repeat_interleave(self.pos_embedding, b, dim=0).permute(0,2,1).reshape(b, self.dim, self.res, self.res)

        source_grid = torch.rand([b, 8, 8, 2], device=x.device) * 2 - 1
        target_grid = torch.rand([b, 8, 8, 2], device=x.device) * 2 - 1
        
        source_feat = F.grid_sample(x, source_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1) # [b, 256, c]

        target_feat = F.grid_sample(x, target_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1) # [b, 256, c]
        target_pos  = F.grid_sample(pe, target_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1) # [b, 256, c]
        
        

        W = F.cosine_similarity(source_feat, target_feat, dim=-1)
        W[W<0] = 0
        
        predictor_input = source_feat + target_pos
        predictor_output = self.predictor(predictor_input) # [b, 256, c]
        
        loss = torch.mean(W * (1 - torch.cosine_similarity(predictor_output, target_feat, dim=-1)))
        return loss
    
    
    def forward_test(self, x, source_grid, target_grid):
        b, n, c = x.shape
        x = self.norm(x)
        x = x.permute(0,2,1).reshape(b,c,self.res,self.res)   
        pe = torch.repeat_interleave(self.pos_embedding, b, dim=0).permute(0,2,1).reshape(b, c, self.res, self.res)
        
        source_feat = F.grid_sample(x, source_grid, mode='bilinear', align_corners=True, padding_mode="border").flatten(2,3).permute(0,2,1) # [b, 256, c]
                
        target_feat = F.grid_sample(x, target_grid, mode='bilinear', align_corners=True, padding_mode="border").flatten(2,3).permute(0,2,1) # [b, 256, c]
        target_pos = F.grid_sample(pe, target_grid, mode='bilinear', align_corners=True, padding_mode="border").flatten(2,3).permute(0,2,1) # [b, 256, c]
        
        predictor_input = source_feat + target_pos
        predictor_output = self.predictor(predictor_input)
        return torch.cosine_similarity(predictor_output, target_feat, dim=-1), source_feat, target_feat
    
    def forward_whole(self, x):
        b, n, c = x.shape
        x = x.permute(0,2,1).reshape(b,c, self.res, self.res)
        
        pe = torch.repeat_interleave(self.pos_embedding, b, dim=0).permute(0,2,1).reshape(b, 384, self.res, self.res)

        source_grid = torch.rand([b, 20, 20, 2], device=x.device) * 2 - 1
        target_grid = torch.rand([b, 20, 20, 2], device=x.device) * 2 - 1
        
        source_feat = F.grid_sample(x, source_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1) # [b, 256, c]
        source_pos = F.grid_sample(pe, source_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1) # [b, 256, c]
        
        target_feat = F.grid_sample(x, target_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1) # [b, 256, c]
        target_pos = F.grid_sample(pe, target_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1) # [b, 256, c]
        
        target_pos = target_pos + source_feat
        target_pos = self.predictor(target_pos) # [b, 256, c]
            
        source_pos = source_pos + target_feat
        source_pos = self.predictor(source_pos) # [b, 256, c]
            
        pred_sim_1 = torch.cosine_similarity(target_pos, target_feat, dim=-1)
        
        pred_sim_2 = torch.cosine_similarity(source_pos, source_feat, dim=-1)
        
        pred_sim_1[pred_sim_1 > pred_sim_2] = pred_sim_2[pred_sim_1 > pred_sim_2]
        
        return pred_sim_1, source_grid, target_grid
    