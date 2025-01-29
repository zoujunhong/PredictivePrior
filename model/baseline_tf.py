from utils.GansformerGenerator import Generator
from utils.networks import ConvNorm, LinearNorm, SlotAttention
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from utils.dino import vit_small
from utils.PredictivePrior import DINOPredictor

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution=224, num_slots=11, num_iterations=3, hid_dim=384):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.slot_dim = 128

        self.encoder = vit_small(patch_size=8)
        
        self.slot_attn = SlotAttention(self.slot_dim, self.slot_dim*4, feat_size=self.hid_dim)
        self.generator = Generator(slot_dim=self.slot_dim, base_dim=512, block_num=4)
        


    def forward(self, image, dino_feat, sigma=0):
        feat = self.encoder(image)
        b,n,c = feat.shape
        slots, _ = self.slot_attn(feat, sigma=sigma)
        
        rec_rgb, attn = self.generator(slots)
        return rec_rgb

if __name__ == '__main__':
    model = SlotAttentionAutoEncoder()
    x = torch.randn([1,3,224,224])
    from thop import profile
    Flops, Params = profile(model,(x,))
    print('Flops:{:6f}G'.format(Flops/1e9))
    print('Params:{:6f}M'.format(Params/1e6))