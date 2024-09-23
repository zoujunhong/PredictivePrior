from .GansformerGenerator import Generator
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from .dino import vit_small

class BasicBlock(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 planes,
                 upsample=False):
        super(BasicBlock, self).__init__()
        self.planes = planes
        self.norm1 = nn.GroupNorm(1, planes)
        self.norm2 = nn.GroupNorm(1, planes)

        self.conv1 = nn.Conv2d(planes, planes, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)

        self.relu = nn.GELU()
        self.upsample = nn.Sequential(
            nn.Conv2d(planes, planes//2, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)) if upsample else nn.Identity()

    def forward(self, x): # slot shape [b*n,c], x shape [b*n,c,h,w]
        """Forward function."""

        def _inner_forward(x):
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            return self.upsample(out + x)

        out = _inner_forward(x)
        out = self.relu(out)
        return out
    
def load_and_freeze(model: nn.Module, dict_name):
    dict = torch.load(dict_name, map_location='cpu')
    model.load_state_dict(dict, strict=True)
    stop_grad(model)

def stop_grad(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = False

def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    
    m = nn.Linear(in_features, out_features, bias)
    
    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    
    if bias:
        nn.init.zeros_(m.bias)
    
    return m


def gru_cell(input_size, hidden_size, bias=True):
    
    m = nn.GRUCell(input_size, hidden_size, bias)
    
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    
    return m


class SlotAttention(nn.Module):
    def __init__(
        self,
        slot_size, 
        mlp_size, 
        feat_size,
        num_slots=11,
        epsilon=1e-6,
    ):
        super().__init__()
        self.slot_size = slot_size 
        self.epsilon = epsilon
        self.num_iters = 3

        self.norm_feature = nn.LayerNorm(feat_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        
        self.project_k = linear(feat_size, slot_size, bias=False)
        self.project_v = linear(feat_size, slot_size, bias=False)
        self.project_q = linear(slot_size, slot_size, bias=False)

        self.slots_init = nn.Embedding(num_slots, slot_size)
        nn.init.xavier_uniform_(self.slots_init.weight)

        self.gru = gru_cell(slot_size, slot_size)

        self.mlp = nn.Sequential(
            linear(slot_size, mlp_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_size, slot_size))

    def forward(self, features, sigma):
        B = features.shape[0]
        mu = self.slots_init.weight.expand(B, -1, -1)
        z = torch.randn_like(mu).type_as(features)
        slots_init = mu + z * sigma * mu.detach()
        # `feature` has shape [batch_size, num_feature, inputs_size].
        features = self.norm_feature(features)  
        k = self.project_k(features)  # Shape: [B, num_features, slot_size]
        v = self.project_v(features)  # Shape: [B, num_features, slot_size]

        B, N, D = v.shape
        slots = slots_init
        # Multiple rounds of attention.
        for i in range(self.num_iters):
            if i == self.num_iters - 1:
                slots = slots.detach() + slots_init - slots_init.detach()
                
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)
            # Attention
            scale = D ** -0.5
            attn_logits= torch.einsum('bid,bjd->bij', q, k) * scale
            attn = F.softmax(attn_logits, dim=1)

            # Weighted mean
            attn_sum = torch.sum(attn, dim=-1, keepdim=True) + self.epsilon
            attn_wm = attn / attn_sum 
            updates = torch.einsum('bij, bjd->bid', attn_wm, v)            

            # Update slots
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            )
            slots = slots.reshape(B, -1, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn

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
    def __init__(self, resolution=224, num_slots=11, num_iterations=3, hid_dim=192):
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

        self.norm = nn.LayerNorm(self.slot_dim, elementwise_affine=False)


    def forward(self, image, sigma=0):
        # encoder
        # with torch.no_grad():
        feat = self.encoder(image)  # CNN Backbone.
        slots, _ = self.slot_attn(feat, sigma=sigma)
        # slots = self.norm(slots)
        rec_rgb, attn = self.generator(slots)
        attn = attn.squeeze(1).permute(0,2,1).reshape(-1,self.num_slots,self.resolution//4,self.resolution//4)
        attn = F.interpolate(attn, size=(224,224), mode='bicubic', align_corners=False)
        # return rec_rgb, attn.reshape(-1,self.num_slots,14,14)
        return rec_rgb, attn
    
    def forward_test(self, image, ignore=-1):
        # encoder
        with torch.no_grad():
            feat = self.encoder(image) # + dino_feat
        b,n,c = feat.shape
        slots,_ = self.slot_attn(feat, sigma=0)
        rec_rgb, attn = self.generator(slots)

        return slots, rec_rgb, attn.permute(0,1,3,2).reshape(-1,11,56,56)