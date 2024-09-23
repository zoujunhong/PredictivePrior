import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import lpips

from taming.modules.losses.vqperceptual import *
from taming.modules.discriminator.model import NLayerDiscriminator

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

class RecLPIPSLoss(nn.Module):

    def __init__(self, percept_loss_w=1.0):
        super().__init__()

        self.perceptual_weight = percept_loss_w
        if self.perceptual_weight > 0.:
            self.perceptual_loss = lpips.LPIPS(net='vgg').eval()
            for p in self.perceptual_loss.parameters():
                p.requires_grad = False

    def forward(self, recon, x):
        x = x.contiguous()
        recon = recon.contiguous()
        recon_loss = F.l1_loss(recon, x)
        percept_loss = self.perceptual_loss(x, recon).mean()
        
        loss_dict = {
            'recon_loss': recon_loss,
            'percept_loss': percept_loss,
            'total_loss': recon_loss + self.perceptual_weight * percept_loss
        }
        return loss_dict


class RecLPIPSLoss_w_Discriminator(nn.Module):

    def __init__(self, percept_loss_w=1.0):
        super().__init__()

        self.perceptual_weight = percept_loss_w
        if self.perceptual_weight > 0.:
            self.perceptual_loss = lpips.LPIPS(net='vgg').eval()
            for p in self.perceptual_loss.parameters():
                p.requires_grad = False
        
        self.disc_loss = hinge_d_loss
        
        self.discriminator = NLayerDiscriminator(input_nc=3,
                                                 n_layers=4,
                                                 use_actnorm=False,
                                                 ndf=32
                                                 ).apply(weights_init)

    def forward(self, recon, image, recon_composition, optimizer_idx, tau):
        image = image.contiguous()
        recon = recon.contiguous()
        
        if optimizer_idx == 0:
            recon_loss = F.l1_loss(recon, image)
            percept_loss = self.perceptual_loss(image, recon).mean()
            logits_fake = self.discriminator(recon_composition.contiguous())
            g_loss = -torch.mean(logits_fake)
            loss_dict = {
                'recon_loss': recon_loss,
                'percept_loss': percept_loss,
                'g_loss': g_loss,
                'total_loss': recon_loss + self.perceptual_weight * percept_loss + tau * g_loss
            }
            return loss_dict
        
        elif optimizer_idx == 1:
            logits_real = self.discriminator(recon.contiguous().detach())
            logits_fake = self.discriminator(recon_composition.contiguous().detach())
            d_loss = self.disc_loss(logits_real, logits_fake)
            return d_loss


def seg_loss(pred, gt): # pred shape [k, h, w], gt shape [h, w]
    with torch.no_grad():
        h,w = gt.shape
        pred_hard = torch.argmax(pred, dim=0)
        temp = 1
        while True:
            if torch.sum(gt == temp) == 0:
                gt[gt > temp] -= 1
            elif torch.sum(gt == temp) > 0:
                temp += 1
            if temp > torch.max(gt):
                break
        
        c1 = torch.max(gt) + 1
        c2 = torch.max(pred_hard) + 1
        
        pred_hard = pred_hard.reshape(-1)
        gt = gt.reshape(-1)

        IOU = torch.zeros([max(c1,c2), max(c1,c2)]).to(pred_hard.device)
        for i in range(c1):
            for j in range(c2):
                I = torch.sum((gt == i) * (pred_hard == j))
                U = torch.sum((gt == i) + (pred_hard == j))
                IOU[i,j] = I / U
        
        IOU_np = (IOU.cpu().numpy() * 100000).astype(np.int32)
        km = KMMatch(IOU_np)
        match = km.match()
        
        gt_match = gt.clone()
        for i in range(match.shape[0]):
            if match[i] == 0:
                gt_match[gt == match[i]] = 255
            else:
                gt_match[gt == match[i]] = i
    
    loss = F.nll_loss(torch.log(pred.unsqueeze(0)+1e-6), gt_match.reshape(h,w).unsqueeze(0), ignore_index=255)
    return 0 if torch.isnan(loss) else loss



# Kuhn-Munkres匹配算法
class KMMatch(object):

    def __init__(self, graph):
        assert isinstance(graph, np.ndarray), print("二分图的必须采用numpy array 格式")
        assert graph.ndim == 2, print("二分图的维度必须为2")
        self.graph = graph

        rows, cols = graph.shape
        self.rows = rows
        self.cols = cols

        self.lx = np.zeros(self.cols, dtype=np.float32)  # 横向结点的顶标
        self.ly = np.zeros(self.rows, dtype=np.float32)  # 竖向结点的顶标

        self.match_index = np.ones(cols, dtype=np.int32) * -1  # 横向结点匹配的竖向结点的index （默认-1，表示未匹配任何竖向结点）
        self.match_weight = 0  # 匹配边的权值之和

        self.inc = math.inf

    def match(self):
        # 初始化顶标, lx初始化为0，ly初始化为节点对应权值最大边的权值
        for y in range(self.rows):
            self.ly[y] = max(self.graph[y, :])

        for y in range(self.rows):  # 从每一竖向结点开始，寻找增广路
            while True:
                self.inc = np.inf
                self.vx = np.zeros(self.cols, dtype=np.int32)  # 横向结点的匹配标志
                self.vy = np.zeros(self.rows, dtype=np.int32)  # 竖向结点的匹配标志
                if self.dfs(y):
                    break
                else:
                    self.update()
                # print(y, self.lx, self.ly, self.vx, self.vy)
        return self.match_index

    # 更新顶标
    def update(self):
        for x in range(self.cols):
            if self.vx[x]:
                self.lx[x] += self.inc
        for y in range(self.rows):
            if self.vy[y]:
                self.ly[y] -= self.inc

    def dfs(self, y):  # 递归版深度优先搜索
        self.vy[y] = 1
        for x in range(self.cols):
            if self.vx[x] == 0:
                t = self.lx[x] + self.ly[y] - self.graph[y][x]
                if t == 0:
                    self.vx[x] = 1
                    # 两种情况：一是结点x没有匹配，那么找到一条增广路；二是X结点已经匹配，采用DFS，沿着X继续往下走，最后若以未匹配点结束，则也是一条增广路
                    if self.match_index[x] == -1 or self.dfs(self.match_index[x]):
                        self.match_index[x] = y  # 未匹配边变成匹配边
                        # print(y, x, self.match_index)
                        return True
                else:
                    if self.inc > t:
                        self.inc = t
        return False