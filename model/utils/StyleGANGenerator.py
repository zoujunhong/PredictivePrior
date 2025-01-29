import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

class LayerNorm2D(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 dim,
                 affine=True):
        super(LayerNorm2D, self).__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=affine)

    def forward(self, x): # x shape [b,c,h,w]
        b,c,h,w = x.shape
        x = x.flatten(2,3).permute(0,2,1)
        x = self.norm(x)
        x = x.permute(0,2,1).reshape(b,c,h,w)
        return x

class ConvNorm(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 inplanes,
                 outplanes,
                 kernel=1,
                 stride=1,
                 padding=0,
                 affine=True,
                 act=True):
        super(ConvNorm, self).__init__()
        self.norm = LayerNorm2D(outplanes, affine=affine)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel, stride, padding)
        self.relu = nn.GELU()
        self.act = act

    def forward(self, x):
        x = self.norm(self.conv(x))
        return self.relu(x) if self.act else x
    
class BasicBlock(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 slot_dim,
                 planes,
                 upsample=False):
        super(BasicBlock, self).__init__()
        self.planes = planes
        self.style = nn.Sequential(
            nn.Linear(slot_dim, 6*planes),
            nn.LayerNorm(6*planes),
            nn.GELU(),
            nn.Linear(6*planes, 6*planes))
        self.norm1 = nn.InstanceNorm2d(planes)
        self.norm2 = nn.InstanceNorm2d(planes)
        self.norm3 = nn.InstanceNorm2d(planes)

        self.conv1 = nn.Conv2d(planes, planes, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)

        self.relu = nn.GELU()
        self.upsample = nn.Sequential(
            nn.Conv2d(planes, planes//2, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ) if upsample else nn.Identity()

    def forward(self, x, slot): # slot shape [b*n,c], x shape [b*n,c,h,w]
        """Forward function."""
        gain1, bias1, gain2, bias2, gain3, bias3 = torch.chunk(self.style(slot), chunks=6, dim = -1)
        out = self.conv1(x)
        out = (1 + gain1[:,:,None,None]) * self.norm1(out) + bias1[:,:,None,None]
        out = self.relu(out)

        out = self.conv2(out)
        out = (1 + gain2[:,:,None,None]) * self.norm2(out) + bias2[:,:,None,None]
        
        out = self.relu(x + out)
        out = (1 + gain3[:,:,None,None]) * self.norm3(out) + bias3[:,:,None,None]
        return self.upsample(out)

class Decoder(nn.Module):
    def __init__(self, slot_dim=64, hid_dim=256, out_dim=3, resolution=8, block_num=4):
        super().__init__()
        self.resolution = resolution
        self.grid = nn.Parameter(torch.randn(1, hid_dim, self.resolution, self.resolution), requires_grad=True) # [1,4,h,w]
        
        self.generator_blocks = nn.ModuleList()
        for i in range(block_num):
            self.generator_blocks.append(BasicBlock(slot_dim=slot_dim, planes=hid_dim, upsample=True))
            hid_dim = hid_dim // 2
            
        self.end = nn.Conv2d(hid_dim, out_dim+1, 1, 1, 0)


    def forward(self, slots):
        B, K, _ = slots.shape
        slots = slots.flatten(0,1)
        x = torch.repeat_interleave(self.grid,B*K,0)
        for i in range(len(self.generator_blocks)):
            x = self.generator_blocks[i](x, slots)
        
        x = self.end(x)
        _, D, H, W = x.shape
        x = x.reshape(B, K, D, H, W)
        recons, masks = torch.split(x, [3,1], dim=2)
        masks = F.softmax(masks, dim=1)
        rec_obj = torch.sum(recons*masks, dim=1)
        return rec_obj, recons, masks.squeeze(2)

