from .utils.networks import SlotAttention, ConvNorm, LinearNorm, load_and_freeze
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from .utils.resnet import ResNet
from .utils.PredictivePrior import DINOPredictor
from .utils.StyleGANGenerator import Decoder

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = nn.Parameter(build_grid(resolution),requires_grad=True)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution=128, num_slots=11, num_iterations=3, hid_dim=64):
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
        self.slot_dim = 64

        self.encoder = ResNet(depth=34)
        self.encoder_pos = SoftPositionEmbed(hid_dim, [self.resolution//4,self.resolution//4])
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, 4 * hid_dim),
            nn.LayerNorm(4 * hid_dim),
            nn.GELU(),
            nn.Linear(4 * hid_dim, hid_dim),
            nn.LayerNorm(hid_dim))
        
        self.pixel_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            ConvNorm(self.hid_dim, self.hid_dim//2, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            ConvNorm(self.hid_dim//2, self.hid_dim//4, 3, 1, 1),
            nn.Conv2d(self.hid_dim//4, self.hid_dim//4, 1, 1, 0))
        
        self.slot_mapping = nn.Sequential(
            LinearNorm(self.slot_dim, self.slot_dim),
            nn.Linear(self.slot_dim, self.hid_dim//4))
        
        self.slot_attn = SlotAttention(self.slot_dim, self.slot_dim*4, feat_size=self.hid_dim, num_slots=self.num_slots)
        self.generator = Decoder(slot_dim=self.slot_dim, hid_dim=256, resolution=self.resolution//16, block_num=4)

        self.predictor = DINOPredictor()
        load_and_freeze(self.predictor, '/root/to/predictor')


    def forward(self, image, dino_feat=torch.randn((1, 256, 384)), sigma=0):
        feat = self.encoder(image)
        b, c, h, w = feat.shape
        feat = feat.permute(0,2,3,1).contiguous()
        print(feat.shape)
        feat = self.encoder_pos(feat)
        feat = torch.flatten(feat, 1, 2)
        feat = feat + self.mlp(feat)  # CNN Backbone.
        
        slots, _ = self.slot_attn(feat, sigma=sigma)
        
        feats_lowrank = self.pixel_decoder(feat.permute(0,2,1).reshape(b, self.hid_dim, self.resolution//4, self.resolution//4))[:,None]
        slots_map = self.slot_mapping(slots)[:,:,:,None,None]
        masks = torch.softmax(torch.sum(feats_lowrank * slots_map, dim=2)/(self.hid_dim/4)**0.5, dim=1)
        
        rec_rgb, recons, alpha_masks = self.generator(slots)

        with torch.no_grad():
            pred_sim, source_grid, target_grid = self.predictor.forward_whole(dino_feat)
            W = ((pred_sim-0.9) * 10).clamp(-1, 1)
            
        source_mask = F.grid_sample(masks, source_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1)
        source_mask = F.normalize(source_mask, dim=-1) # [b, 256, k]
        
        target_mask = F.grid_sample(masks, target_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1)
        target_mask = F.normalize(target_mask, dim=-1)
        
        delta = 1 - torch.cosine_similarity(source_mask, target_mask, dim=-1)
        
        loss_smooth = torch.mean(W * delta) + F.l1_loss(alpha_masks, masks.detach())
        return rec_rgb, loss_smooth
    

if __name__ == '__main__':
    model = SlotAttentionAutoEncoder()
    x = torch.randn([1,3,128,128])
    from thop import profile
    Flops, Params = profile(model,(x,))
    print('Flops:{:6f}G'.format(Flops/1e9))
    print('Params:{:6f}M'.format(Params/1e6))