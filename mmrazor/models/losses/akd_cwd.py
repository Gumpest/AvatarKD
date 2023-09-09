# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class AKD_ChannelWiseDivergence(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.

    <https://arxiv.org/abs/2011.13256>`_.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau=1.0,
        loss_weight=1.0,
        drop_rate=0.1
    ):
        super(AKD_ChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

        self.Uncertain = nn.BatchNorm2d(2048, affine=False)
        self.drop_rate = drop_rate

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        if not isinstance(y_s, (tuple, list)):
            y_s = (y_s, )
            y_t = (y_t, )
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape

            N, C, H, W = s.shape

            # 1. multi-tea
            multi_T = F.dropout(t, p=self.drop_rate, training=True)

            # 2. uncertainty weights
            s = self.Uncertain(s)
            t = self.Uncertain(t)

            # normalize in channel diemension
            softmax_pred_T = F.softmax(multi_T.view(-1, W * H) / self.tau, dim=1) # [N*C, H*W]

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(softmax_pred_T *
                            logsoftmax(multi_T.view(-1, W * H) / self.tau) -
                            softmax_pred_T *
                            logsoftmax(s.view(-1, W * H) / self.tau)) * (
                                self.tau**2)
            
            losses.append(cost / (C * N))
        loss = sum(losses)

        return loss * self.loss_weight
