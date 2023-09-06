# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrazor.registry import MODELS

# def feature_vis(inputs,times):
#     batch_size = inputs[0].shape[0]
#
#     for batch in range(batch_size):
#         import mmrazor.utils.misc
#         feat = mmrazor.utils.misc.show_featmaps(inputs[batch])
#         import mmcv
#         feat_rgb = mmcv.bgr2rgb(feat)
#         mmcv.imwrite(feat_rgb, f'./output/feat_rgb_{times}.jpg')

@MODELS.register_module()
class UncerLoss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.75
    """

    def __init__(self,
                 student_channels,
                 teacher_channels,
                 inputs_channels,
                 name='loss_uncer',
                 alpha_mgd=0.00002,
                 # lambda_mgd=0.75,
                 # flag1=True,
                 # flag2=True,
                 a=1,
                 ):
        super(UncerLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        # self.lambda_mgd = lambda_mgd
        self.name = name
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)
        self.a = a
        self.times = 0

        # if student_channels != teacher_channels:
        #     self.align1 = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        # else:
        #     self.align1 = None

        self.align1 = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        self.align2 = nn.Conv2d(student_channels, inputs_channels, kernel_size=1, stride=1, padding=0)

        self.maxpool = nn.MaxPool2d(8)

        self.generation1 = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))

        self.generation2 = nn.Sequential(
            nn.Conv2d(student_channels, student_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(student_channels, student_channels, kernel_size=3, padding=1))

    def forward(self,
                preds_S,
                preds_T,
                inputs = None):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        # if self.align is not None:
        #     preds_S = self.align1(preds_S)

        # loss = self.get_dis_loss(preds_S, preds_T) * self.alpha_mgd

        # batch_size = inputs[0].shape[0]
        #
        # for batch in range(batch_size):
        #     import mmrazor.utils.misc
        #     ori_img = mmrazor.utils.misc.show_featmaps(inputs[0][batch], is_image=True, show=False)
        #     v = self.get_var_loss(inputs, preds_S)
        #     feat = mmrazor.utils.misc.show_featmaps(v[batch])
        #     import mmcv
        #     feat_rgb = mmcv.bgr2rgb(feat)
        #     # feat_bgr = mmcv.rgb2bgr(feat)
        #     mmcv.imwrite(ori_img, f'./output3/img_{self.times}.jpg')
        #     # mmcv.imwrite(feat, f'./output/feat_{self.times}.jpg')
        #     mmcv.imwrite(feat_rgb, f'./output3/feat_rgb_{self.times}.jpg')
        #     # mmcv.imwrite(feat_bgr, f'./output/feat_bgr_{self.times}.jpg')
        #     self.times += 1

        # for batch in range(batch_size):
        #     import mmrazor.utils.misc
        #     ori_img = mmrazor.utils.misc.show_featmaps(inputs[0][batch], is_image=True, show=False)
        #     v = self.get_var_loss(inputs, preds_S)
        #     feat = mmrazor.utils.misc.show_featmaps(v[batch])
        #     import mmcv
        #     mmcv.imwrite(ori_img, f'./output1/img_{self.times}.jpg')
        #     mmcv.imwrite(feat, f'./output1/feat_{self.times}.jpg')
        #     self.times += 1

        # for batch in range(batch_size):
        #     import mmrazor.utils.misc
        #     ori_img = mmrazor.utils.misc.show_featmaps(inputs[0][batch], is_image=True, show=False)
        #     v = self.get_var_loss(inputs, preds_S)
        #     from mmseg.models.utils.wrappers import resize
        #     v = resize(
        #         input=v,
        #         size=inputs[0].shape[2:],
        #         mode='bilinear',
        #         align_corners=False)
        #     feat = v[batch]
        #
        #     feat = feat.detach().cpu().numpy().transpose(1, 2, 0)
        #     for i in range(3):
        #         f = feat[..., i]
        #         f = (f - f.min()) / (f.max() - f.min())
        #         feat[..., i] = f
        #     feat = feat * 255
        #     import mmcv
        #     mmcv.imwrite(ori_img, f'./output/img_{self.times}.jpg')
        #     mmcv.imwrite(feat, f'./output/feat_{self.times}.jpg')
        #     self.times += 1

        bias = torch.mean(self.get_bias_loss(preds_S, preds_T))
        var = torch.mean(self.get_var_loss(inputs, preds_S))
        # loss = bias / (2 * var) + var
        exp_variance = torch.exp(-var)
        loss = bias * exp_variance + var

        return loss

    def get_bias_loss(self, preds_S, preds_T):

        kl_distance = nn.KLDivLoss(reduction='none')

        preds_S = self.align1(preds_S)

        new_fea1 = self.generation1(preds_S)
        bias_loss = kl_distance(self.log_sm(new_fea1), self.sm(preds_T))

        return bias_loss

    def get_var_loss(self, inputs, preds_S):

        kl_distance = nn.KLDivLoss(reduction='none')
        inputs = inputs[0]

        new_fea2 = self.generation2(preds_S)
        new_fea2 = self.align2(new_fea2)
        inputs = self.maxpool(inputs)

        # inputs_i = inputs[..., 30:54, 45: 59]
        # new_fea2_i = new_fea2[..., 30:54, 45: 59]
        # variance = kl_distance(self.log_sm(inputs_i), self.sm(new_fea2_i))
        # variance_i = torch.mean(variance)

        var_loss = kl_distance(self.log_sm(inputs), self.sm(new_fea2))

        mean_var = torch.mean(var_loss, dim=1)
        conv = nn.Conv2d(1, 1, 5, 5).cuda()
        conv.weight = nn.Parameter(torch.ones_like(conv.weight))
        conv.bias = nn.Parameter(torch.zeros_like(conv.bias))
        conv_m_var = conv(mean_var)


        return var_loss

