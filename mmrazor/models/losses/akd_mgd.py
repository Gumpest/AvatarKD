# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
import math

class MaskModule(nn.Module):
    def __init__(self, channels):
        super(MaskModule, self).__init__()
        self.generation = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(channels, channels, kernel_size=3, padding=1))

    def forward(self, x):

        return self.generation(x)

@LOSSES.register_module()
class AKD_MGDLoss(nn.Module):
    def __init__(self, alpha_mgd=0.00002, lambda_mgd=0.65, teacher_channels=[2048], drop_rate=0.1):
        super(AKD_MGDLoss, self).__init__()

        self.mask_modules = nn.ModuleList([
            MaskModule(channels=c) for c in teacher_channels]
        )
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.Uncertain = nn.ModuleList([
            nn.BatchNorm2d(c, affine=False) for c in teacher_channels]
        )
        self.drop_rate = drop_rate

    def forward(self,
                y_s_list,
                y_t_list,):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        if not isinstance(y_s_list, (tuple, list)):
            y_s_list = (y_s_list, )
            y_t_list = (y_t_list, )
        assert len(y_s_list) == len(y_t_list) == len(self.mask_modules)

        losses = []
        for idx, (preds_S, preds_T) in enumerate(zip(y_s_list, y_t_list)):
            assert preds_S.shape[-2:] == preds_T.shape[-2:]
            # 1. multi-tea
            preds_T = F.dropout(preds_T, p=self.drop_rate, training=True)

            # 2. uncertainty weights           
            preds_S = self.Uncertain[idx](preds_S)
            preds_T = self.Uncertain[idx](preds_T)

            loss = self.get_dis_loss(preds_S, preds_T, idx)
            losses.append(loss)

        loss = sum(losses)
        return loss * self.alpha_mgd

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat > 1-self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.mask_modules[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T)/N

        return dis_loss