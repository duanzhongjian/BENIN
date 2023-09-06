# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
from mmcls.models import VisionTransformer
from mmseg.models.backbones import VisionTransformer
from mmseg.models.backbones import MixVisionTransformer
from mmselfsup.registry import MODELS
from ..utils import build_2d_sincos_position_embedding


@MODELS.register_module()
class MAEViT(MixVisionTransformer):
    def __init__(self,
                 embed_dims=64,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 out_indices: Union[Sequence, int] = -1,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None,
                 mask_ratio: float = 0.75):
        super().__init__(
            embed_dims=embed_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            patch_sizes=patch_sizes,
            out_indices=out_indices,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)

        self.layers[0].patch_embed.requires_grad = False
        self.mask_ratio = mask_ratio

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        outs = []

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs
