# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
from mmengine.model import BaseModule
from torch import nn

from mmselfsup.registry import MODELS
from ..utils import Encoder


@MODELS.register_module()
class CAEHead(BaseModule):
    """Pretrain Head for CAE.

    Compute the align loss and the main loss. In addition, this head also
    generates the prediction target generated by dalle.

    Args:
        loss (dict): The config of loss.
        tokenizer_path (str): The path of the tokenizer.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 loss: dict,
                 tokenizer_path: str,
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.tokenizer_path = tokenizer_path
        self.encoder = self._load_encoder()
        self.loss = MODELS.build(loss)

    def _load_encoder(self) -> nn.Module:
        """Load the dalle to generate target features."""
        encoder = Encoder()
        if os.path.exists(self.tokenizer_path):
            state_dict = torch.load(self.tokenizer_path)
            encoder.load_state_dict(state_dict)
        else:
            warnings.warn(
                f'Do not find {self.tokenizer_path}, please download from https://download.openmmlab.com/mmselfsup/cae/dalle_encoder.pth'  # noqa: E501
            )
        return encoder

    @torch.no_grad()
    def _generate_target(self, img_target: torch.Tensor) -> torch.Tensor:
        """Generate the reconstruction target."""
        logits = self.encoder(img_target)
        target = torch.argmax(logits, dim=1)
        return target.flatten(1)

    def forward(self, logits: torch.Tensor, img_target: torch.Tensor,
                latent_pred: torch.Tensor, latent_target: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate loss.

        Args:
            logits (torch.Tensor): Logits generated by decoder.
            img_target (img_target): Target generated by dalle for decoder
                prediction.
            latent_pred (torch.Tensor): Latent prediction by regressor.
            latent_target (torch.Tensor): Target for latent prediction,
                generated by teacher.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The tuple of loss.
                - loss_main (torch.Tensor): Cross entropy loss.
                - loss_align (torch.Tensor): MSE loss.
        """

        target = self._generate_target(img_target)  # target features
        target = target[mask].detach()

        # loss main for decoder, loss align for regressor
        loss_main, loss_align = self.loss(logits, target, latent_pred,
                                          latent_target)

        return (loss_main, loss_align)