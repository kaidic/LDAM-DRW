import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


class VSLoss(nn.Module):

    def __init__(self, min_cls_list, n_cls=10, iota_pos=0.0, iota_neg=0.0, Delta_pos=1.0, Delta_neg=1.0,
                 device=torch.device('cuda'), weight=None):

        super(VSLoss, self).__init__()
        iota_list, Delta_list = iota_neg * torch.ones(n_cls), Delta_neg * torch.ones(n_cls)
        iota_list[min_cls_list], Delta_list[min_cls_list] = iota_pos, Delta_pos

        self.device = device
        self.iota_list = iota_list.to(device)
        self.Delta_list = Delta_list.to(device)
        self.weight = weight

    def forward(self, x, target):

        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor).to(self.device)

        batch_iota = torch.matmul(self.iota_list, index_float.transpose(0, 1))
        batch_Delta = torch.matmul(self.Delta_list, index_float.transpose(0, 1))
        batch_iota = batch_iota.view((-1, 1))
        batch_Delta = batch_Delta.view((-1, 1))

        x_d = x * batch_Delta
        x_di = x_d - batch_iota

        output = torch.where(index, x_di, x_d)

        return F.cross_entropy(output, target, weight=self.weight)