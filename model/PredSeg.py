from .networks import ConvNorm, LinearNorm
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from .resnet import ResNet
from .predictor import DINOPredictor
from .StyleGANGenerator import Decoder
# from .invariant_sa import SlotAttention as ISA

def load_and_freeze(model: nn.Module, dict_name):
    dict = torch.load(dict_name, map_location='cpu',weights_only=True)
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


def gru_cell(input_size, hidden_size):
    
    m = nn.GRUCell(input_size, hidden_size)
    
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    
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

        self.gru = nn.GRUCell(self.slot_size, self.slot_size) # gru_cell(slot_size, slot_size)

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
    def __init__(self, resolution=224, num_slots=11, num_iterations=3, hid_dim=64):
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
        
        # self.pixel_decoder = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
        #     ConvNorm(self.hid_dim, self.hid_dim//2, 3, 1, 1),
        #     nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
        #     ConvNorm(self.hid_dim//2, self.hid_dim//4, 3, 1, 1),
        #     nn.Conv2d(self.hid_dim//4, self.hid_dim//4, 1, 1, 0))
        
        # self.slot_mapping = nn.Sequential(
        #     LinearNorm(self.slot_dim, self.slot_dim),
        #     nn.Linear(self.slot_dim, self.hid_dim//4))
        
        self.slot_attn = SlotAttention(self.slot_dim, self.slot_dim*4, feat_size=self.hid_dim, num_slots=self.num_slots)
        self.generator = Decoder(slot_dim=self.slot_dim, hid_dim=256, resolution=self.resolution//16, target_resolution=self.resolution, block_num=4)

        # self.slot_attn = ISA()
        self.predictor = DINOPredictor()
        load_and_freeze(self.predictor, '/root/onethingai-tmp/savemodel/PTR_predictor_mae/model_epoch_1000.pth')


    def forward(self, image, dino_feat, sigma=0):
        feat = self.encoder(image)
        b, c, h, w = feat.shape
        feat = feat.permute(0,2,3,1).contiguous()
        feat = self.encoder_pos(feat)
        feat = torch.flatten(feat, 1, 2)
        feat = feat + self.mlp(feat)  # CNN Backbone.
        slots, masks = self.slot_attn(feat, sigma=sigma)
        masks = masks.reshape(b, self.num_slots, 32, 32)
        # attns = F.interpolate(attns, scale_factor=4, mode='bilinear', align_corners=False)
        
        # feats_lowrank = self.pixel_decoder(feat.permute(0,2,1).reshape(b, self.hid_dim, self.resolution//4, self.resolution//4))[:,None]
        # slots_map = self.slot_mapping(slots)[:,:,:,None,None]
        # masks = torch.softmax(torch.sum(feats_lowrank * slots_map, dim=2)/(self.hid_dim/4)**0.5, dim=1)
        
        slots_bc = slots.reshape((-1, slots.shape[-1]))
        temp = self.generator(slots_bc)
        recons, alpha_masks = temp.reshape(b, -1, temp.shape[1], temp.shape[2], temp.shape[3]).split([3,1], dim=2)
        alpha_masks = F.softmax(alpha_masks, dim=1)
        rec_rgb = torch.sum(recons * alpha_masks, dim=1)  # Recombine image.
        # attns = attns.squeeze(2)

        with torch.no_grad():
            pred_sim, source_grid, target_grid = self.predictor.forward_whole(dino_feat)
            W = ((pred_sim-0.9) * 30).clamp(-1, 1)
            
        source_mask = F.grid_sample(masks, source_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1)
        source_mask = F.normalize(source_mask, dim=-1) # [b, 256, k]
        
        target_mask = F.grid_sample(masks, target_grid, mode='bilinear', align_corners=True).flatten(2,3).permute(0,2,1)
        target_mask = F.normalize(target_mask, dim=-1)
        
        delta = 1 - torch.cosine_similarity(source_mask, target_mask, dim=-1)
        
        loss_smooth = torch.mean(W * delta) #+ F.l1_loss(attns, masks.detach())
        return rec_rgb, loss_smooth
    
    def forward_test(self, image, ignore=[]):
        feat = self.encoder(image)
        b, c, h, w = feat.shape
        feat = feat.permute(0,2,3,1).contiguous()
        feat = self.encoder_pos(feat)
        feat = torch.flatten(feat, 1, 2)
        feat = feat + self.mlp(feat)  # CNN Backbone.
        b,n,c = feat.shape
        slots, _ = self.slot_attn(feat, sigma=0)
        # masks = masks.reshape(b, self.num_slots, 28, 28)
        
        feats_lowrank = self.pixel_decoder(feat.permute(0,2,1).reshape(b, self.hid_dim, self.resolution//4, self.resolution//4))[:,None]
        slots_map = self.slot_mapping(slots)[:,:,:,None,None]
        masks = torch.softmax(torch.sum(feats_lowrank * slots_map, dim=2), dim=1)
        
        # slots = self.norm(slots)
        if len(ignore) > 0:
            for idx in ignore:
                slots = torch.cat([slots[:,:idx], slots[:,idx+1:]], dim=1)
                
        slots_bc = slots.reshape((-1, slots.shape[-1]))
        temp = self.generator(slots_bc)
        recons,attns = temp.reshape(b, -1, temp.shape[1], temp.shape[2], temp.shape[3]).split([3,1], dim=2)
        attns = F.softmax(attns, dim=1)
        rec_rgb = torch.sum(recons * attns, dim=1)  # Recombine image.
        attns = attns.squeeze(2)
        # attn = F.interpolate(attn, scale_factor=4, mode='bilinear', align_corners=False)
        # print(masks.shape)
        
        return slots, rec_rgb, attns
    

if __name__ == '__main__':
    model = SlotAttentionAutoEncoder()
    x = torch.randn([1,3,224,224])
    from thop import profile
    Flops, Params = profile(model,(x,))
    print('Flops:{:6f}G'.format(Flops/1e9))
    print('Params:{:6f}M'.format(Params/1e6))