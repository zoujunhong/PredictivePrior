import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.utils.networks import HardSoftmax
from scipy.optimize import linear_sum_assignment

def ARI(pred, gt, ignore = 100): # pred shape [h, w], gt shape [h, w]
    
    c1 = torch.max(gt) + 1
    c2 = torch.max(pred) + 1
    # print(c1,c2)
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    
    valid = (gt != ignore).long()
    gt = gt[valid != 0]
    pred = pred[valid != 0]
    len = gt.shape[0]
    # print(len,c1,c2)
    with torch.no_grad():
        n = torch.zeros([len, c1*c2]).to(pred.device)
        index = (gt * c2 + pred).unsqueeze(-1).long()
        src = torch.ones([len,c1*c2]).to(pred.device)
        n.scatter_(1, index, src)

        n = n.reshape(len,c1,c2).sum(dim=0)
        
        a = torch.sum(n, dim=0)
        b = torch.sum(n, dim=1)

        RI = torch.sum(n * (n-1))
        ERI = torch.sum(a * (a-1)) * torch.sum(b * (b-1)) / (len * (len - 1))
        maxRI = 0.5 * (torch.sum(a * (a-1)) + torch.sum(b * (b-1)))

        ARI = (RI - ERI + 1e-8) / (maxRI - ERI + 1e-8)
        # if torch.isnan(ARI):
        #     print(RI, ERI, maxRI)
        return ARI


def MIOU(pred, gt, ignore = 0): # pred shape [h, w], gt shape [h, w]
    
    c1 = torch.max(gt) + 1
    c2 = torch.max(pred) + 1
    # print(c1,c2)
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    # valid = (gt != ignore).long()
    # gt = gt[valid != 0]
    # gt -= 1
    # pred = pred[valid != 0]
    IOU = torch.zeros([max(c1,c2), max(c1,c2)]).to(pred.device)
    start = 1 if ignore==0 else 0
    for i in range(start,c1):
        for j in range(c2):
            I = torch.sum((gt == i) * (pred == j))
            U = torch.sum((gt == i) + (pred == j))
            IOU[i,j] = I / U
    # print(IOU)
    indices = linear_sum_assignment(1 - IOU.squeeze(0).cpu().numpy())

    IOU_list = []
    Pix_list = []
    OIOU = 0
    # print(match.shape)
    for i in range(max(c1,c2)):
        OIOU += IOU[i, indices[1][i]]
        IOU_list.append(IOU[indices[1][i],i].item())
        Pix_list.append(torch.sum(gt == indices[1][i]).item())
    
    return OIOU/(c1-1), c1, IOU_list, Pix_list

def MBO(pred, gt, ignore = 100): # pred shape [h, w], gt shape [h, w]
    c1 = torch.max(gt)+1
    c2 = torch.max(pred) + 1
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    IOU = torch.zeros([c1, c2]).to(pred.device)
    start = 1 if ignore==0 else 0
    count_0 = 0
    for i in range(start,c1):
        if torch.sum(gt == i) > 0:
            count_0+=1
        for j in range(c2):
            I = (gt == i) * (pred == j)
            I = torch.sum(I)
            U = (gt == i) + (pred == j)
            U = torch.sum(U)
            IOU[i,j] = I / (U+1e-8)
    
    IOU_max = torch.max(IOU, dim=-1)[0]
    MSC = (torch.sum(IOU_max)+1e-4) / (count_0+1e-4)

    return MSC
