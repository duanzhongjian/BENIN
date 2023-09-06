# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule
from mmselfsup.registry import MODELS
import numpy as np
from ..utils import resize
from mmcv.cnn import ConvModule

@MODELS.register_module()
class MGHead(BaseModule):
    """Head for contrastive learning.

    The contrastive loss is implemented in this head and is used in SimCLR,
    MoCo, DenseCL, etc.

    Args:
        loss (dict): Config dict for module of loss functions.
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Defaults to 0.1.
    """

    def __init__(self, loss: dict,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 n_class=3) -> None:
        super().__init__()

        self.loss = MODELS.build(loss)
        self.n_class = n_class
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_fcg = act_cfg
        self.conv = ConvModule(512,
                               3,
                               kernel_size=3,
                               padding=1,
                               dilation=1,
                               conv_cfg=self.conv_cfg,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_fcg)

    def logits(self, x):
        """Get the logits before the cross_entropy loss.

        This module is used to obtain the logits before the loss.

        Args:
            x (List[Tensor] | Tuple[Tensor]): Feature maps of backbone,
                each tensor has shape (N, C, H, W).

        Returns:
            List[Tensor]: A list of class scores.
        """
        x = self.conv(x)
        return x

    def forward(self, x: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Forward function to compute contrastive loss.

        Args:
            pos (torch.Tensor): Nx1 positive similarity.
            neg (torch.Tensor): Nxk negative similarity.

        Returns:
            torch.Tensor: The contrastive loss.
        """

        logit = self.logits(x[0])
        mse = torch.nn.MSELoss()
        b, c, w, h = inputs.shape
        logit = resize(
            input=logit,
            size=inputs.shape[2:],
            mode='bilinear',
            align_corners=False)

        loss = mse(logit, inputs)
        return loss

