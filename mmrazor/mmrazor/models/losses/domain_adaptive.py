# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrazor.registry import MODELS


@MODELS.register_module()
class DasLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.75
    """

    def __init__(self,
                 name='loss_das',
                 alpha=0.00002,
                 tau=1.0
                 ):
        super(DasLoss, self).__init__()
        self.alpha = alpha
        self.name = name
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)
        self.tau = tau
        self.times = 0

    def forward(self,
                # inputs,
                preds_S,
                preds_T,
                preds_Aux,):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        preds_S = preds_S / self.tau
        preds_T = preds_T / self.tau

        # if self.align is not None:
        #     preds_S = self.align1(preds_S)

        # loss = self.get_dis_loss(preds_S, preds_T) * self.alpha_mgd

        # batch_size = inputs[0].shape[0]
        #
        # for batch in range(batch_size):
        #     import mmrazor.utils.misc
        #     ori_img = mmrazor.utils.misc.show_featmaps(inputs[0][batch], is_image=True, show=False)
        #     v = self.get_var_loss(preds_S, preds_Aux)
        #     feat = mmrazor.utils.misc.show_featmaps(v[batch])
        #     import mmcv
        #     feat_rgb = mmcv.bgr2rgb(feat)
        #     # feat_bgr = mmcv.rgb2bgr(feat)
        #     mmcv.imwrite(ori_img, f'./output/img_{self.times}.jpg')
        #     # mmcv.imwrite(feat, f'./output/feat_{self.times}.jpg')
        #     mmcv.imwrite(feat_rgb, f'./output/feat_rgb_{self.times}.jpg')
        #     # mmcv.imwrite(feat_bgr, f'./output/feat_bgr_{self.times}.jpg')
        #     self.times += 1

        # kl_distance = nn.KLDivLoss(reduction='none')
        # preds_Aux_i = preds_Aux[..., 26:33, 23: 29]
        # preds_S_i = preds_S[..., 26:33, 23: 29]
        # variance = kl_distance(self.log_sm(preds_Aux_i), self.sm(preds_S_i))
        # variance_i = torch.mean(variance)

        bias = torch.mean(self.get_bias_loss(preds_S, preds_T))
        var = torch.mean(self.get_var_loss(preds_Aux, preds_S))
        exp_variance = torch.exp(-var)
        loss = bias * exp_variance + var
        loss = (self.tau**2) * loss
        loss = loss * self.alpha

        return loss

    def get_bias_loss(self, preds_S, preds_T):

        criterion = nn.CrossEntropyLoss(reduction='none')
        bias_loss = criterion(self.log_sm(preds_S), self.sm(preds_T))

        return bias_loss

    def get_var_loss(self, preds_S, preds_Aux):

        kl_distance = nn.KLDivLoss(reduction='none')
        var_loss = kl_distance(self.log_sm(preds_Aux), self.sm(preds_S))

        mean_var = torch.mean(var_loss, dim=1)
        conv = nn.Conv2d(1, 1, 5, 5).cuda()
        conv.weight = nn.Parameter(torch.ones_like(conv.weight))
        conv.bias = nn.Parameter(torch.zeros_like(conv.bias))
        conv_m_var = conv(mean_var)

        return var_loss

