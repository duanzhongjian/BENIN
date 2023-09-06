# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmengine.structures import LabelData

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel
from ..utils.mg_utils import *

@MODELS.register_module()
class ModelGenesis(BaseModel):

    def extract_feat(self, inputs: List[torch.Tensor],
                     **kwargs) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            inputs (List[torch.Tensor]): The input images.

        Returns:
            Tuple[torch.Tensor]: Backbone outputs.
        """

        x = self.backbone(inputs)
        return x

    def loss(self, inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """

        # image deformation
        nonlinear_rate = 0.9
        paint_rate = 0.9
        outpaint_rate = 0.8
        inpaint_rate = 1.0 - outpaint_rate
        local_rate = 0.5
        # flip_rate = 0.4

        x = inputs[0]
        for n in range(inputs[0].shape[0]):
            # Autoencoder
            x[n] = copy.deepcopy(inputs[0][n])

            # Local Shuffle Pixel
            x[n] = local_pixel_shuffling(x[n], prob=local_rate)

            # Apply non-Linear transformation with an assigned probability
            x[n] = nonlinear_transformation(x[n], nonlinear_rate)

            #Inpainting & Outpainting
            if random.random() < paint_rate:
                if random.random() < inpaint_rate:
                    # Inpainting
                    x[n] = image_in_painting(x[n])
                else:
                    # Outpainting
                    x[n] = image_out_painting(x[n])

        x = self.backbone(x)
        loss = self.head(x, inputs[0])
        losses = dict(loss=loss)
        return losses

    def predict(self, inputs: List[torch.Tensor],
                data_samples: List[SelfSupDataSample],
                **kwargs) -> List[SelfSupDataSample]:
        """The forward function in testing.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            List[SelfSupDataSample]: The prediction from model.
        """
        x = self.backbone(inputs)
        outs = self.head.logits(x)

        for i in range(len(outs)):
            prediction_data = {key: out[i] for key, out in zip(keys, outs)}
            prediction = LabelData(**prediction_data)
            data_samples[i].pred_label = prediction

        return data_samples
