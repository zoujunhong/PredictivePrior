import numpy as np
from torch import nn
import torch
from .networks import TransformerEncoderLayer, TransformerDecoderLayer
    
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

class BasicBlock_tf(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 slot_dim,
                 planes):
        super(BasicBlock_tf, self).__init__()
        self.planes = planes
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(slot_dim, 6*planes),
            nn.LayerNorm(6*planes),
            nn.GELU(),
            nn.Linear(6*planes, 6*planes))
        self.norm1 = nn.LayerNorm(planes)
        self.norm2 = nn.LayerNorm(planes)
        self.attn = TransformerEncoderLayer(planes, planes//32, 0)

    def modulate(self, x, shift, scale):
        return x * (1 + scale[:,None,:]) + shift[:,None,:]
    
    def forward(self, x, slots): # slot shape [B*N,C], x shape [B*N,L,C]
        """Forward function."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(slots).chunk(6, dim=1)
        x = x + gate_msa[:,None,:] * self.attn.attn(self.modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp[:,None,:] * self.attn.ffn(self.modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).permute(0,3,1,2).contiguous()

class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.embedding = nn.Conv2d(4, hidden_size, 1, 1, 0, bias=True)
        self.grid = nn.Parameter(build_grid(resolution),requires_grad=True)

    def forward(self):
        grid = self.embedding(self.grid)
        return grid

class Decoder(nn.Module):
    def __init__(self, slot_dim=384, hid_dim=256, out_dim=3, resolution=8, block_num=4, target_resolution=128):
        super().__init__()
        assert 2**block_num * resolution >= target_resolution
        self.grid = nn.Parameter(build_grid([resolution, resolution]), requires_grad=True) # [1,4,h,w]
        self.embedding = nn.Conv2d(4, hid_dim, 1, 1, 0)
        self.generator_blocks = nn.ModuleList()
        for i in range(block_num):
            upsample = resolution < target_resolution
            self.generator_blocks.append(BasicBlock(slot_dim, hid_dim, upsample=upsample))
            hid_dim = hid_dim//2 if upsample else hid_dim
            resolution = resolution*2 if upsample else resolution
        
        # self.end_cnn = nn.Conv2d(hid_dim, out_dim+1, 1, 1, 0)
        self.end_cnn = nn.Sequential(
            nn.Conv2d(hid_dim, max(hid_dim, out_dim+1), 3, 1, 1),
            LayerNorm2D(max(hid_dim, out_dim+1)),
            nn.GELU(),
            nn.Conv2d(max(hid_dim, out_dim+1), out_dim+1, 1, 1, 0))

    def forward(self, slots):
        b = slots.shape[0]
        init_grid = torch.repeat_interleave(self.embedding(self.grid),b,0)
        for i in range(len(self.generator_blocks)):
            init_grid = self.generator_blocks[i](init_grid, slots)

        return self.end_cnn(init_grid)


class Decoder_tf(nn.Module):
    def __init__(self, slot_dim=384, hid_dim=192, out_dim=3, resolution=14, block_num=6):
        super().__init__()
        self.resolution = resolution // 16
        self.grid = nn.Parameter(torch.randn(1,self.resolution**2, hid_dim), requires_grad=True) # [1,4,h,w]
        
        self.generator_blocks = nn.ModuleList()
        for i in range(block_num):
            self.generator_blocks.append(BasicBlock_tf(slot_dim=slot_dim, planes=hid_dim))
            
        self.end_cnn = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            ConvNorm(hid_dim, hid_dim//2, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            ConvNorm(hid_dim//2, hid_dim//4, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            ConvNorm(hid_dim//4, hid_dim//8, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            ConvNorm(hid_dim//8, hid_dim//16, 3, 1, 1),
            nn.Conv2d(hid_dim//16, out_dim+1, 1, 1, 0),)


    def forward(self, slots):
        b = slots.shape[0]
        x = torch.repeat_interleave(self.grid,b,0)
        for i in range(len(self.generator_blocks)):
            x = self.generator_blocks[i](x, slots)
        
        x = x.permute(0,2,1).reshape(b, -1, self.resolution, self.resolution)
        return self.end_cnn(x)

