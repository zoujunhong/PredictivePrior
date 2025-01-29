from .utils.GansformerGenerator import Generator
from .utils.networks import SlotAttention, ConvNorm, LinearNorm, HardSoftmax, load_and_freeze, stop_grad
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from .utils.dino import vit_small
from .utils.PredictivePrior import DINOPredictor

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
        
        self.dino = vit_small(patch_size=8)
        to_load = torch.load('/path/to/dino_deitsmall8_pretrain.pth',map_location=torch.device("cpu"))
        self.dino.load_state_dict(to_load,strict=True)
        stop_grad(self.dino)

        self.encoder = vit_small(patch_size=8)
        
        self.pixel_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            ConvNorm(self.hid_dim, self.hid_dim//3, 3, 1, 1),
            nn.Conv2d(self.hid_dim//3, self.hid_dim//3, 1, 1, 0))
        
        self.slot_mapping = nn.Sequential(
            LinearNorm(self.slot_dim, self.slot_dim),
            nn.Linear(self.slot_dim, self.hid_dim//3))
        
        self.slot_attn = SlotAttention(self.slot_dim, self.slot_dim*4, feat_size=self.hid_dim)
        self.generator = Generator(slot_dim=self.slot_dim, base_dim=512, block_num=4)
        
        self.predictor = DINOPredictor()
        to_load = torch.load('/path/to/predictor',map_location=torch.device("cpu"))
        self.predictor.load_state_dict(to_load,strict=True)
        stop_grad(self.predictor)

    def forward(self, image, dino_feat, sigma=0):
        feat = self.encoder(image)
        b,n,c = feat.shape
        slots, _ = self.slot_attn(feat, sigma=sigma)
        
        feats_lowrank = self.pixel_decoder(feat.permute(0,2,1).reshape(b, self.hid_dim, self.resolution//8, self.resolution//8))[:,None]
        slots_map = self.slot_mapping(slots)
        masks = torch.softmax(torch.sum(feats_lowrank * slots_map[:,:,:,None,None], dim=2)/(self.hid_dim/3)**0.5, dim=1)
        
        rec_rgb, attn = self.generator(slots)
        attn = attn.squeeze(1).permute(0,2,1).reshape(b,-1,self.resolution//4,self.resolution//4)

        with torch.no_grad():
            dino_feat = self.dino.forward_key(image)
            pred_sim, source_grid, target_grid = self.predictor.forward_whole(dino_feat)
            W = ((pred_sim-0.3) * 10).clamp(-1, 1)
            
        source_mask = F.grid_sample(masks, source_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1)
        source_mask = F.normalize(source_mask, dim=-1) # [b, 256, k]
        
        target_mask = F.grid_sample(masks, target_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1)
        target_mask = F.normalize(target_mask, dim=-1)
        
        delta = 1 - torch.cosine_similarity(source_mask, target_mask, dim=-1)
        
        loss_seg = torch.mean(W * delta)
        loss_consistent = F.l1_loss(attn, masks.detach())
        return rec_rgb, loss_seg, loss_consistent
    
    def forward_test(self, image, ignore=[]):
        feat = self.encoder(image) # + dino_feat
        b,n,c = feat.shape
        slots, _ = self.slot_attn(feat, sigma=0)
        # masks = masks.reshape(b, self.num_slots, 28, 28)
        
        feats_lowrank = self.pixel_decoder(feat.permute(0,2,1).reshape(b, self.hid_dim, self.resolution//8, self.resolution//8))[:,None]
        slots_map = self.slot_mapping(slots)[:,:,:,None,None]
        masks = torch.softmax(torch.sum(feats_lowrank * slots_map, dim=2), dim=1)
        
        # slots = self.norm(slots)
        if len(ignore) > 0:
            for idx in ignore:
                slots[:,idx] = 0.
                # slots = torch.cat([slots[:,:idx], slots[:,idx+1:]], dim=1)
        
        rec_rgb, attn = self.generator(slots)
        attn = attn.squeeze(1).permute(0,2,1).reshape(b,-1,self.resolution//4,self.resolution//4)
        attn = F.interpolate(attn, scale_factor=4, mode='bilinear', align_corners=False)
        # print(masks.shape)
        # masks = F.interpolate(masks, scale_factor=8, mode='bilinear', align_corners=False)
        
        return slots, rec_rgb, attn
    

if __name__ == '__main__':
    model = SlotAttentionAutoEncoder()
    x = torch.randn([1,3,224,224])
    from thop import profile
    Flops, Params = profile(model,(x,))
    print('Flops:{:6f}G'.format(Flops/1e9))
    print('Params:{:6f}M'.format(Params/1e6))