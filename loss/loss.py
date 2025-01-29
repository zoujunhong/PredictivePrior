import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import lpips

from taming.modules.losses.vqperceptual import *
from taming.modules.discriminator.model import NLayerDiscriminator

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
