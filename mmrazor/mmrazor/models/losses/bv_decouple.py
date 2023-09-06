# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn

from mmrazor.registry import MODELS


@MODELS.register_module()
class BVDKDLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation.
    <https://arxiv.org/abs/2205.01529>`
    Args:
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
    """

    def __init__(self, alpha_mgd: float = 0.00002, a=1) -> None:
        super(BVDKDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)
        self.a = a
        self.loss_mse = nn.MSELoss(reduction='none')
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.kl_distance = nn.KLDivLoss(reduction='none')
        self.L1loss = nn.L1Loss(reduction='none')

    def forward(self, preds_S: torch.Tensor,
                preds_T: torch.Tensor,
                new_fea: torch.Tensor) -> torch.Tensor:
        """Forward function.
        Args:
            preds_S(torch.Tensor): Bs*C*H*W, student's feature map
            preds_T(torch.Tensor): Bs*C*H*W, teacher's feature map
        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape == preds_T.shape
        loss = self.get_dis_loss(preds_S, preds_T, new_fea) * self.alpha_mgd

        return loss

    def get_dis_loss(self, preds_S: torch.Tensor,
                     preds_T: torch.Tensor,
                     new_fea: torch.Tensor) -> torch.Tensor:
        """Get MSE distance of preds_S and preds_T.
        Args:
            preds_S(torch.Tensor): Bs*C*H*W, student's feature map
            preds_T(torch.Tensor): Bs*C*H*W, teacher's feature map
        Return:
            torch.Tensor: The calculated mse distance value.
        """
        N, C, H, W = preds_T.shape
        dis_loss = self.loss_mse(preds_S, preds_T) / N

        # loss = self.criterion(new_fea, preds_T)
        # variance = torch.sum(self.kl_distance(self.log_sm(new_fea), self.sm(preds_S)), dim=1)
        #
        # loss = self.kl_distance(self.log_sm(new_fea), self.sm(preds_T))
        # variance = self.kl_distance(self.log_sm(new_fea), self.sm(preds_S))
        #
        loss = self.loss_mse(self.log_sm(new_fea), self.sm(preds_T))
        variance = self.loss_mse(self.log_sm(new_fea), self.sm(preds_S))
        #
        # loss = self.L1loss(self.log_sm(new_fea), self.sm(preds_T))
        # variance = self.L1loss(self.log_sm(new_fea), self.sm(preds_S))
        #
        exp_variance = torch.exp(-variance)

        dis_loss1 = torch.mean(self.a * loss + (1-self.a) * loss * exp_variance) + torch.mean(variance)

        dis_loss = dis_loss1/N

        return dis_loss