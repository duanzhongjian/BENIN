# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import mmcv
from mmrazor.registry import MODELS
import mmrazor.utils.misc

# def feature_vis(inputs, a, times):
#     batch_size = inputs[0].shape[0]
#
#     for batch in range(batch_size):
#         import mmrazor.utils.misc
#         ori_img = mmrazor.utils.misc.show_featmaps(inputs[0][batch], is_image=True, show=False)
#         feat = mmrazor.utils.misc.show_featmaps(a[batch])
#         import mmcv
#         feat_rgb = mmcv.rgb2bgr(feat)
#         mmcv.imwrite(ori_img, f'./output2/imgS_{times}.jpg')
#         mmcv.imwrite(feat_rgb, f'./output2/preS_bgr_{times}.jpg')

@MODELS.register_module()
class Featureloss(nn.Module):
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
                 name='loss_benin',
                 alpha_mgd=0.00002,
                 lambda_mgd=0.75,
                 with_inputs=False,
                 # flag1=True,
                 # flag2=True,
                 a=1,
                 ):
        super(Featureloss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        self.name = name
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)
        # self.flag1 = flag1
        # self.flag2 = flag2
        self.a = a
        self.times = 0
        self.with_inputs = with_inputs

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))

    def forward(self,
                preds_S,
                preds_T,
                inputs):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]

        if self.align is not None:
            preds_S = self.align(preds_S)
        # self.inputs = inputs
        loss = self.get_dis_loss(inputs, preds_S, preds_T) * self.alpha_mgd


        # self.feature_vis(inputs, preds_T, name='preds_T')
        # self.feature_vis(inputs, preds_S, name='preds_S')

        return loss

    # def get_dis_loss(self, preds_S, preds_T):
    #     loss_mse = nn.MSELoss(reduction='sum')
    #     N, C, H, W = preds_T.shape
    #
    #     device = preds_S.device
    #     mat = torch.rand((N,1,H,W)).to(device)
    #     mat = torch.where(mat>1-self.lambda_mgd, 0, 1).to(device)
    #
    #     masked_fea = torch.mul(preds_S, mat)
    #     new_fea = self.generation(masked_fea)
    #
    #     dis_loss = loss_mse(new_fea, preds_T)/N
    #
    #     return dis_loss

    # def feature_vis(self, inputs, b, name):
    #     batch_size = b.shape[0]
    #
    #     for batch in range(batch_size):
    #         input = inputs[batch]
    #         file_name = input.img_path
    #         ori_img = mmrazor.utils.misc.show_featmaps(mmcv.imread(file_name), is_image=True, show=False)
    #         file_name = file_name.split('/')[-1].split('.')[0]
    #         feat = mmrazor.utils.misc.show_featmaps(b[batch], channel_reduction='select_max')
    #         feat_rgb = mmcv.rgb2bgr(feat)
    #         mmcv.imwrite(ori_img, f'./outputT/{file_name}/img.jpg')
    #         mmcv.imwrite(feat_rgb, f'./outputT/{file_name}/{name}.jpg')

    def get_dis_loss(self, inputs, preds_S, preds_T):

        # loss_mse = nn.MSELoss(reduction='none')
        # criterion = nn.CrossEntropyLoss(reduction = 'none')
        kl_distance = nn.KLDivLoss(reduction='none')
        # L1loss = nn.L1Loss(reduction='none')

        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N,C,H,W)).to(device)
        mat = torch.where(mat>1-self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        # self.feature_vis(self.inputs, masked_fea, name='masked_fea')
        # self.feature_vis(self.inputs, new_fea, name='new_fea')

        # batch_size = inputs[0].shape[0]
        #
        # for batch in range(batch_size):
        #     import mmrazor.utils.misc
        #     ori_img = mmrazor.utils.misc.show_featmaps(inputs[0][batch], is_image=True, show=False)
        #     v = kl_distance(self.log_sm(new_fea), self.sm(preds_S))
        #     feat = mmrazor.utils.misc.show_featmaps(v[batch])
        #     import mmcv
        #     feat_rgb = mmcv.bgr2rgb(feat)
        #     feat_bgr = mmcv.rgb2bgr(feat)
        #     mmcv.imwrite(ori_img, f'./output2/img_{self.times}.jpg')
        #     mmcv.imwrite(feat, f'./output2/feat_{self.times}.jpg')
        #     mmcv.imwrite(feat_rgb, f'./output2/feat_rgb_{self.times}.jpg')
        #     mmcv.imwrite(feat_bgr, f'./output2/feat_bgr_{self.times}.jpg')
        #     self.times += 1

        # loss = criterion(new_fea, preds_T)
        # variance = torch.sum(kl_distance(self.log_sm(new_fea), self.sm(preds_S)), dim=1)

        loss = kl_distance(self.log_sm(new_fea), self.sm(preds_T))
        variance = kl_distance(self.log_sm(new_fea), self.sm(preds_S))

        # var_loss = kl_distance(self.log_sm(new_fea), self.sm(preds_S))
        # mean_var = torch.mean(var_loss, dim=1)
        # conv = nn.Conv2d(1, 1, 5, 5).cuda()
        # conv.weight = nn.Parameter(torch.ones_like(conv.weight))
        # conv.bias = nn.Parameter(torch.zeros_like(conv.bias))
        # conv_m_var = conv(mean_var)

        # new_fea_i = new_fea[..., 30:54, 45: 59]
        # preds_S_i = preds_S[..., 30:54, 45: 59]
        # variance = kl_distance(self.log_sm(new_fea_i), self.sm(preds_S_i))
        # variance_i = torch.mean(variance)

        # loss = loss_mse(self.log_sm(new_fea), self.sm(preds_T))
        # variance = loss_mse(self.log_sm(new_fea), self.sm(preds_S))

        # loss = L1loss(self.log_sm(new_fea), self.sm(preds_T))
        # variance = L1loss(self.log_sm(new_fea), self.sm(preds_S))

        exp_variance = torch.exp(-variance)

        dis_loss1 = torch.mean(self.a * loss + (1-self.a) * loss * exp_variance) + torch.mean(variance)

        dis_loss = dis_loss1/N

        return dis_loss

    # def get_dis_loss(self, preds_S, preds_T):
    #     # loss_mse = nn.MSELoss(reduction='none')
    #
    #     # criterion = nn.CrossEntropyLoss(reduction='none')
    #     kl_distance = nn.KLDivLoss(reduction='none')
    #
    #     N, C, H, W = preds_T.shape
    #
    #     device = preds_S.device
    #     mat = torch.rand((N, C, H, W)).to(device)
    #     mat = torch.where(mat >1-self.lambda_mgd, 0, 1).to(device)
    #
    #     masked_fea = torch.mul(preds_S, mat)
    #     new_fea = self.generation(masked_fea)
    #
    #     # loss = loss_mse(new_fea, preds_T)
    #
    #     loss = kl_distance(self.log_sm(new_fea), self.sm(preds_T))
    #     variance = kl_distance(self.log_sm(new_fea), self.sm(preds_S))
    #
    #     # loss = criterion(new_fea, preds_T)
    #     # variance = torch.sum(kl_distance(self.log_sm(new_fea), self.sm(preds_S)), dim=1)
    #     exp_variance = torch.exp(-variance)
    #     dis_loss1 = torch.mean(loss * exp_variance) + torch.mean(variance)
    #     elif self.flag2 is True & self.flag1 is False:
    #           dis_loss1 = torch.mean(a * loss + (1-a) * loss * exp_variance) + torch.mean(variance)
    #     elif self.flag2 is False & self.flag1 is False:
    #           dis_loss1 = torch.mean(loss * exp_variance) + torch.mean(variance)
    #     dis_loss = dis_loss1 / N
    #
    #     return dis_loss