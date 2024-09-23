import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
def load_and_freeze(model: nn.Module, dict_name):
    dict = torch.load(dict_name, map_location='cpu')
    model.load_state_dict(dict, strict=True)
    stop_grad(model)

def stop_grad(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = False
        
class LayerNorm2D(nn.Module):
    """Basic block for ResNet."""
    def __init__(self,
                 dim,
                 affine=True):
        super(LayerNorm2D, self).__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=affine)

    def forward(self, x: torch.Tensor): # x shape [b,c,h,w]
        b,c,h,w = x.shape
        x = x.flatten(2,3).permute(0,2,1).contiguous()
        x = self.norm(x)
        x = x.permute(0,2,1).reshape(b,c,h,w).contiguous()
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

def farthest_point_sample(xyz, npoint): 

    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    centroids = torch.zeros((B, npoint), dtype=torch.long).to(device)   # 采样点矩阵（B, npoint）
    distance = torch.ones(B, N).to(device) * 1e10                       # 采样点到所有点距离（B, N）

    batch_indices = torch.arange(B, dtype=torch.long).to(device)        # batch_size 数组
    
    barycenter = torch.mean(xyz, dim=1, keepdim=True)                   #计算重心坐标 及 距离重心最远的点

    dist = torch.sum((xyz - barycenter) ** 2, dim=-1)
    farthest = torch.argmax(dist, -1)                                   #将距离重心最远的点作为第一个点
    sampled_points = []
    for i in range(npoint):
        centroids[:, i] = farthest                                      # 更新第i个最远点
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)        # 取出这个最远点的xyz坐标
        sampled_points.append(centroid)
        dist = torch.sum((xyz - centroid) ** 2, -1)                     # 计算点集中的所有点到这个最远点的欧式距离
        distance[dist < distance] = dist[dist < distance]               # 更新distance，记录样本中每个点距离所有已出现的采样点的最小距离

        farthest = torch.argmax(distance, -1)                           # 返回最远点索引
    
    return torch.cat(sampled_points, dim=1)

class FFN(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x):
        return x + self.encoder(x)
    
def QuietSoftmax(x, dim=-1):
    x = x - torch.max(x)
    x = torch.exp(x)
    return x / (1 + torch.sum(x, dim=dim, keepdim=True))

def HardSoftmax(x, dim=-1):
    y_soft = F.softmax(x, dim=dim)
    index = y_soft.argmax(dim, keepdim=True)
    y_hard = torch.zeros_like(x).scatter_(dim, index, 1.)
    return (y_hard - y_soft).detach() + y_soft

############################################# Transformer #############################################
# -----------------------------------------------------------------------------------------------------

# Transpose tensor to scores
def transpose_for_scores(x, num_heads, elem_num, head_size):
    x = x.reshape(-1, elem_num, num_heads, head_size).permute(0, 2, 1, 3).contiguous() # [B, N, H, S]
    return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob + 1e-8) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MultiheadAttention(torch.nn.Module):
    def __init__(self,
            output_cap_dim,           input_cap_dim,             # The from/to tensors dimensions
            # Additional options
            num_heads           = 6,                # Number of attention heads
            attention_dropout   = .0,             # Attention dropout rate
            direction = 0,
        ):                             # Ignore unrecognized keyword args

        super().__init__()
        self.to_q = nn.Linear(output_cap_dim, output_cap_dim)
        self.to_k = nn.Linear(input_cap_dim, output_cap_dim)
        self.to_v = nn.Linear(input_cap_dim, output_cap_dim)

        self.dim = output_cap_dim
        self.output_cap_dim = output_cap_dim
        self.to_dim = input_cap_dim
        
        self.num_heads = num_heads
        self.size_head = int(output_cap_dim / num_heads)

        self.norm_input = nn.LayerNorm(input_cap_dim)
        self.norm = nn.LayerNorm(output_cap_dim) 
        self.dropout = DropPath(attention_dropout)

        self.proj = nn.Linear(output_cap_dim, output_cap_dim)
        
        self.norm_direction = direction


    def forward(self, input, output, mask=None): # mask shape [B, output_num, input_num]
        # queries, keys and values
        i = self.norm_input(input)
        o = self.norm(output)
        queries = self.to_q(o)
        keys    = self.to_k(i)
        values  = self.to_v(i)
        # Reshape queries, keys and values, and then compute att_scores
        b, n1, c1 = input.shape
        b, n2, c2 = output.shape
        values  = transpose_for_scores(values,  self.num_heads, n1,   self.size_head)  # [B, N, T, H]
        queries = transpose_for_scores(queries, self.num_heads, n2,   self.size_head)  # [B, N, F, H]
        keys    = transpose_for_scores(keys,    self.num_heads, n1,   self.size_head)  # [B, N, T, H]
        att_scores = torch.matmul(queries, keys.transpose(-1, -2)) / self.size_head ** 0.5 # [B,N,output_num,input_num]
        if mask is not None:
            att_scores = torch.masked_fill(att_scores, mask, float('-inf'))
        
        att_probs = F.softmax(att_scores, dim=-1)

        # Compute weighted-sum of the values using the attention distribution
        control = att_probs.matmul(values)      # [B, N, F, H]
        control = control.permute(0, 2, 1, 3)   # [B, F, N, H]
        b, n, h, d = control.shape
        control = control.reshape(b, n, self.dim) # [B*F, N*H]
        # This newly computed information will control the bias/gain of the new from_tensor
        output = output + self.dropout(self.proj(control))
        return queries, output


class SimplexAttention(torch.nn.Module):
    def __init__(self,
            output_dim,           
            input_dim,
            # Additional options
            num_heads           = 6,              # Number of attention heads
            attention_dropout   = 0,              # Attention dropout rate
            direction = 0
        ):

        super().__init__()
        self.to_q = nn.Linear(output_dim, output_dim)
        self.to_k = nn.Linear(input_dim, output_dim)
        self.to_v = nn.Linear(input_dim, output_dim)

        self.dim = output_dim
        self.output_dim = output_dim
        self.to_dim = input_dim
        
        self.num_heads = num_heads
        self.size_head = int(output_dim / num_heads)

        self.norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        # self.dropout = DropPath(attention_dropout)

        self.modulation = nn.Sequential(
            nn.Linear(output_dim, 2*output_dim),
            nn.GELU(),
            nn.Linear(2*output_dim, 2*output_dim)
        )

        self.proj = nn.Linear(output_dim, output_dim)
        self.norm_direction = direction
    

    def integrate(self, tensor, control): # integration, norm
        # Normalize tensor
        tensor = self.norm(tensor)
        # Compute gain/bias
        control = self.modulation(control)
        gain, bias = torch.split(control, [self.output_dim, self.output_dim], dim = -1)
        tensor = tensor * (gain + 1) + bias
        return tensor


    def forward(self, input, output): # mask shape [B, input_num]
        # queries, keys and values
        queries = self.to_q(output)
        keys    = self.to_k(input)
        values  = self.to_v(input)
        # Reshape queries, keys and values, and then compute att_scores
        b, n1, c1 = input.shape
        b, n2, c2 = output.shape
        queries = transpose_for_scores(queries, self.num_heads, n2, self.size_head)  # [B, N, F, H]
        keys    = transpose_for_scores(keys,    self.num_heads, n1, self.size_head)  # [B, N, T, H]
        values  = transpose_for_scores(values,  self.num_heads, n1, self.size_head)  # [B, N, T, H]

        att_scores = torch.matmul(queries, keys.transpose(-1, -2)) / self.size_head ** 0.5 # [B,N,output_num,input_num]  

        if self.norm_direction == 0:
            att_probs = F.softmax(att_scores, dim=-1)
        else:
            att_probs = F.softmax(att_scores, dim=-2)
            att_probs = att_probs / (att_probs.sum(dim=-1, keepdim=True) + 1)
            
        control = att_probs.matmul(values)      # [B, N, F, H]
        control = control.permute(0, 2, 1, 3)   # [B, F, N, H]
        b, n, h, d = control.shape
        control = control.reshape(b, n, self.dim) # [B*F, N*H]
        # This newly computed information will control the bias/gain of the new from_tensor
        output = self.integrate(output, self.proj(control))
        return att_probs, output


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self,
            output_cap_dim,           input_cap_dim,             # The from/to tensors dimensions
            # Additional options
            num_heads           = 6,                # Number of attention heads
            attention_dropout   = .0,             # Attention dropout rate
            self_attn=True,
            attn_type='vanilla',
            direction=0
        ):                             # Ignore unrecognized keyword args

        super().__init__()
        assert attn_type in ['vanilla', 'simplex']
        self.self_attn = self_attn
        if self_attn:
            self.self_attn = MultiheadAttention(output_cap_dim, output_cap_dim, num_heads, attention_dropout)
        self.multihead_attn = \
            MultiheadAttention(output_cap_dim, input_cap_dim, num_heads, attention_dropout, direction=direction) \
            if attn_type == 'vanilla' else SimplexAttention(output_cap_dim, input_cap_dim, num_heads, attention_dropout, direction=direction)
        self.droppath = DropPath(attention_dropout)
        self.FFN = nn.Sequential(
            nn.LayerNorm(output_cap_dim),
            nn.Linear(output_cap_dim,4*output_cap_dim),
            nn.GELU(),
            nn.Linear(4*output_cap_dim,output_cap_dim))

    def forward(self, input_cap, output_cap, mask_input=None, mask_output=None):
        if self.self_attn:
            _, output_cap = self.self_attn(output_cap, output_cap, mask=mask_output)
            
        attn, output_cap = self.multihead_attn(input_cap, output_cap)
        output_cap = output_cap + self.droppath(self.FFN(output_cap))
        return attn, output_cap

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self,
            output_cap_dim,             # The from/to tensors dimensions
            # Additional options
            num_heads           = 6,                # Number of attention heads
            attention_dropout   = .0,             # Attention dropout rate
            direction = 0
        ):                             # Ignore unrecognized keyword args

        super().__init__()
        self.self_attn = MultiheadAttention(output_cap_dim, output_cap_dim, num_heads, attention_dropout, direction = direction)
        self.droppath = DropPath(attention_dropout)
        self.FFN = nn.Sequential(
            nn.Linear(output_cap_dim,4*output_cap_dim),
            nn.LayerNorm(4*output_cap_dim),
            nn.GELU(),
            nn.Linear(4*output_cap_dim,output_cap_dim),
            nn.LayerNorm(output_cap_dim))


    def forward(self, x, mask=None):
        x, attn = self.attn(x, mask=mask)
        x = self.ffn(x)
        return x, attn
    
    def attn(self, x, mask=None):
        attn, x = self.self_attn(x, x, mask=mask)
        return x, attn
    
    def ffn(self, x):
        x = x + self.droppath(self.FFN(x))
        return x