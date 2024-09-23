import torch
import torch.nn as nn
import torch.nn.functional as F

def PropertyPredLoss(pred_property, category, pos):
    pred_category, pred_pos = pred_property.split([17, 27])
    pred_category = F.log_softmax(pred_category, dim=0)
    category = torch.zeros([17]).to(category.device).scatter_(0, category, 1.)
    loss_category = -torch.sum(pred_category*category)
    loss_pos = F.mse_loss(pred_pos, pos)
    return loss_category, loss_pos

