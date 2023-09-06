import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, build_norm_layer, ConvModule
from ..utils import nlc_to_nchw, nchw_to_nlc
from mmcv.cnn.bricks import HSigmoid
from mmseg.registry.registry import MODELS
import torch.nn.functional as F
class SK_Module(nn.Module):
    def __init__(self, in_channels, conv_cfg=None):
        super(SK_Module, self).__init__()
        if conv_cfg is not None and conv_cfg['type'] == 'Conv3d':
            self.gap = nn.AdaptiveAvgPool3d(1)
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg),
            ConvModule(
                in_channels=in_channels // 2,
                out_channels=in_channels,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg))
        self.softmax = nn.Softmax(1)

    def forward(self, x1, x2):
        u = self.gap(x1 + x2)
        u = self.bottleneck(u)
        softmax_a = self.softmax(u)
        out = x1 * softmax_a + x2 * (1 - softmax_a)
        return out

# @MODELS.register_module()
# class EX_Module(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  with_self=True,
#                  ratio=2,
#                  # channels,
#                  conv_cfg=None,
#                  norm_cfg=dict(type='LN', eps=1e-6)):
#         super(EX_Module, self).__init__()
#         self.in_channels = in_channels
#         self.channels = in_channels // ratio
#         self.with_self = with_self
#         self.conv_q_right = ConvModule(in_channels,
#                                        1,
#                                        kernel_size=1,
#                                        stride=1,
#                                        bias=False,
#                                        conv_cfg=conv_cfg)
#         self.conv_v_right = ConvModule(in_channels,
#                                        self.channels,
#                                        kernel_size=1,
#                                        stride=1,
#                                        bias=False,
#                                        conv_cfg=conv_cfg)
#         # self.ln = nn.LayerNorm(normalized_shape=[self.channels, 1, 1])
#         # self.softmax_right = nn.Softmax(dim=2)
#         # self.sigmoid_right = nn.Sigmoid()
#         self.conv_up = ConvModule(self.channels,
#                                   in_channels,
#                                   kernel_size=1,
#                                   stride=1,
#                                   bias=False,
#                                   # act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0),
#                                   # norm_cfg=norm_cfg,
#                                   conv_cfg=conv_cfg)
#
#         self.conv_q_left = ConvModule(in_channels,
#                                       self.channels,
#                                       kernel_size=1,
#                                       stride=1,
#                                       bias=False,
#                                       conv_cfg=conv_cfg)   #g
#         self.conv_v_left = ConvModule(in_channels,
#                                       self.channels,
#                                       kernel_size=1,
#                                       stride=1,
#                                       bias=False,
#                                       conv_cfg=conv_cfg)   #theta
#         # self.softmax_left = nn.Softmax(dim=1)
#         # self.sigmoid_left = nn.Sigmoid()
#         self.sk = SK_Module(in_channels=in_channels,
#                             conv_cfg=conv_cfg)
#         if with_self is True:
#             # self.softmax_self = nn.Softmax(dim=2)
#             self.resConv = ConvModule(
#                 in_channels,
#                 in_channels,
#                 kernel_size=1,
#                 bias=False,
#                 conv_cfg=conv_cfg)
#     def forward(self, x):
#         """Forward function."""
#         b, c, h, w = x.size()
#
#         # Spatial Attention (psa)
#         input_x = self.conv_v_right(x)  #b, c/2, h, w
#         context_mask = self.conv_q_right(x) #b, 1, h, w
#         context_mask = torch.softmax(context_mask.reshape(b, 1, h * w), dim=2) #b, 1, h*w
#         # context_mask = self.softmax_right(context_mask.reshape(b, 1, h * w)) #b, 1, h*w
#         context_mask = context_mask.reshape(b, 1, h * w)
#         context = torch.matmul(input_x.reshape(b, self.channels, h * w),
#                                context_mask.transpose(1, 2)) #b, C/2, 1
#         # spatial_attn = self.conv_up(F.layer_norm(context.reshape(b, c//2, 1, 1), normalized_shape=(c//2, 1, 1))).reshape(b, c, 1)   #b, c, 1
#         # spatial_attn = torch.sigmoid(F.layer_norm(context.reshape(b, self.channels, 1, 1), normalized_shape=(self.channels, 1, 1)))   #b, c/2, 1
#
#         # spatial_attn = self.ln(context.reshape(b, self.channels, 1, 1))   #b, c/2, 1
#         spatial_attn = context.reshape(b, self.channels, 1, 1)   #b, c/2, 1
#         # spatial_attn = self.sigmoid_right(self.conv_up(spatial_attn).reshape(b, c, 1)) #b, c, 1
#         # spatial_attn = F.layer_norm(self.conv_up(spatial_attn), normalized_shape=(c, 1, 1)).reshape(b, c, 1) #b, c, 1
#         spatial_attn = torch.sigmoid(
#             F.layer_norm(self.conv_up(spatial_attn), normalized_shape=(c, 1, 1))).reshape(b, c, 1)  # b, c/2, 1
#         # spatial_out = x * spatial_attn
#
#         # Channel Attention (psa)
#         g_x = self.conv_q_left(x)   #b, c/2, h, w
#         # avg_x = self.softmax_left(F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, self.channels)) #b, c/2
#         avg_x = torch.softmax(F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, self.channels), dim=1) #b, c/2
#         # avg_x = F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, self.channels)  # b, c/2
#         avg_x = avg_x.reshape(b, 1, self.channels)
#         theta_x = self.conv_v_left(x).reshape(b, self.channels, h * w)  #b, c/2, h*w
#         context = torch.matmul(avg_x, theta_x)  #b, 1, h*w
#         # channel_attn = self.sigmoid_left(context) #b, 1, h*w
#         channel_attn = torch.sigmoid(context) #b, 1, h*w
#         # channel_out = x * channel_attn
#
#         sequence = torch.bmm(spatial_attn, channel_attn).reshape(b, c, h, w)
#
#         parallel = (channel_attn + spatial_attn).reshape(b, c, h, w)
#         # sequence = (x * channel_attn) * spatial_attn
#         #
#         # parallel = x * channel_attn + x * spatial_attn
#         sk_attn = self.sk(sequence, parallel)
#         # sk_attn = torch.cat((sequence, parallel), dim=1)
#
#         if self.with_self is True:
#             self_attn = torch.softmax(sk_attn.reshape(b, c, h * w), dim=2).reshape(b, c, h, w)
#             # self_attn = self.softmax_self(sk_attn.reshape(b, c, h * w)).reshape(b, c, h, w)
#             value = self.resConv(x)
#             self_attn = self_attn * value
#             self_attn = self_attn.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
#             out = self_attn
#         else:
#             out = x * sk_attn
#         return out + x

@MODELS.register_module()
class EX_Module(nn.Module):
    def __init__(self,
                 in_channels,
                 with_self=True,
                 ratio=2,
                 # channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(EX_Module, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // ratio
        self.with_self = with_self
        self.conv_q_right = ConvModule(in_channels,
                                       1,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_v_right = ConvModule(in_channels,
                                       self.channels,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_up = ConvModule(self.channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False,
                                  act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0),
                                  conv_cfg=conv_cfg)

        self.conv_q_left = ConvModule(in_channels,
                                      self.in_channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #g
        self.conv_v_left = ConvModule(in_channels,
                                      self.in_channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #theta
        self.sk = SK_Module(in_channels=in_channels,
                            conv_cfg=conv_cfg)
        if with_self is True:
            self.resConv = ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg)
    def forward(self, x):
        """Forward function."""
        b, c, h, w = x.size()

        # Spatial Attention (psa)
        input_x = self.conv_v_right(x)  #b, c/2, h, w
        context_mask = self.conv_q_right(x) #b, 1, h, w
        context_mask = torch.softmax(context_mask.reshape(b, 1, h * w), dim=2) #b, 1, h*w
        context_mask = context_mask.reshape(b, 1, h * w)
        context = torch.matmul(input_x.reshape(b, self.channels, h * w),
                               context_mask.transpose(1, 2)) #b, C/2, 1
        # spatial_attn = self.conv_up(F.layer_norm(context.reshape(b, c//2, 1, 1), normalized_shape=(c//2, 1, 1))).reshape(b, c, 1)   #b, c, 1
        spatial_attn = torch.sigmoid(F.layer_norm(context.reshape(b, self.channels, 1, 1), normalized_shape=(self.channels, 1, 1)))   #b, c/2, 1
        spatial_attn = self.conv_up(spatial_attn).reshape(b, c, 1) #b, c, 1
        # spatial_out = x * spatial_attn

        # Channel Attention (psa)
        g_x = self.conv_q_left(x)   #b, c/2, h, w
        avg_x = torch.softmax(F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, self.channels), dim=1) #b, c/2
        # avg_x = F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, self.channels)  # b, c/2
        avg_x = avg_x.reshape(b, 1, self.channels)
        theta_x = self.conv_v_left(x).reshape(b, self.channels, h * w)  #b, c/2, h*w
        context = torch.matmul(avg_x, theta_x)  #b, 1, h*w
        channel_attn = torch.sigmoid(context) #b, 1, h*w

        # channel_out = x * channel_attn

        sequence = torch.bmm(spatial_attn, channel_attn).reshape(b, c, h, w)

        parallel = (channel_attn + spatial_attn).reshape(b, c, h, w)
        # sequence = (x * channel_attn) * spatial_attn
        #
        # parallel = x * channel_attn + x * spatial_attn
        sk_attn = self.sk(sequence, parallel)
        # sk_attn = torch.cat((sequence, parallel), dim=1)

        if self.with_self is True:
            self_attn = torch.softmax(sk_attn.reshape(b, c, h * w), dim=2).reshape(b, c, h, w)
            value = self.resConv(x)
            self_attn = self_attn * value
            self_attn = self_attn.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
            out = self_attn
        else:
            out = x * sk_attn
        return out + x

@MODELS.register_module()
class EX_Module_sp(nn.Module):
    def __init__(self,
                 in_channels,
                 ratio=2,
                 with_self=True,
                 # channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(EX_Module_sp , self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // ratio
        self.with_self = with_self
        self.conv_q_right = ConvModule(in_channels,
                                       1,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_v_right = ConvModule(in_channels,
                                       self.channels,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_up = ConvModule(self.channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False,
                                  act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0),
                                  conv_cfg=conv_cfg)

        if with_self is True:
            self.resConv = ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg)
    def forward(self, x):
        """Forward function."""
        b, c, h, w = x.size()

        # Spatial Attention (psa)
        input_x = self.conv_v_right(x)  #b, c/2, h, w
        context_mask = self.conv_q_right(x) #b, 1, h, w
        context_mask = torch.softmax(context_mask.reshape(b, 1, h * w), dim=2) #b, 1, h*w
        context_mask = context_mask.reshape(b, 1, h * w)
        context = torch.matmul(input_x.reshape(b, self.channels, h * w),
                               context_mask.transpose(1, 2)) #b, C/2, 1

        spatial_attn = torch.sigmoid(F.layer_norm(context.reshape(b, self.channels, 1, 1), normalized_shape=(self.channels, 1, 1)))   #b, c/2, 1

        spatial_attn = self.conv_up(spatial_attn) #b, c, 1

        if self.with_self is True:
            value = self.resConv(x)
            self_attn = spatial_attn * value
            self_attn = self_attn.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
            out = self_attn
        else:
            out = x * spatial_attn
        return out + x

@MODELS.register_module()
class EX_Module_ch(nn.Module):
    def __init__(self,
                 in_channels,
                 with_self=True,
                 ratio=1,
                 # channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(EX_Module_ch , self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // ratio
        self.with_self = with_self

        self.conv_q_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #g
        self.conv_v_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #theta

        if with_self is True:
            self.resConv = ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg)
    def forward(self, x):
        """Forward function."""
        b, c, h, w = x.size()

        # Channel Attention (psa)
        g_x = self.conv_q_left(x)   #b, c/r, h, w
        avg_x = torch.softmax(F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, self.channels), dim=1) #b, c/r
        avg_x = avg_x.reshape(b, 1, self.channels)
        theta_x = self.conv_v_left(x).reshape(b, self.channels, h * w)  #b, c/r, h*w
        context = torch.matmul(avg_x, theta_x)  #b, 1, h*w
        channel_attn = torch.sigmoid(context).reshape(b, 1, h, w)  #b, 1, h, w

        if self.with_self is True:
            value = self.resConv(x)
            self_attn = channel_attn * value
            self_attn = self_attn.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
            out = self_attn
        else:
            out = x * channel_attn
        return out + x

@MODELS.register_module()
class EX_Module_sq(nn.Module):
    def __init__(self,
                 in_channels,
                 with_self=True,
                 ratio=2,
                 # channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(EX_Module_sq, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // ratio
        self.with_self = with_self
        self.conv_q_right = ConvModule(in_channels,
                                       1,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_v_right = ConvModule(in_channels,
                                       self.channels,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_up = ConvModule(self.channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False,
                                  act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0),
                                  conv_cfg=conv_cfg)

        self.conv_q_left = ConvModule(in_channels,
                                      self.in_channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #g
        self.conv_v_left = ConvModule(in_channels,
                                      self.in_channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #theta

        if with_self is True:
            self.resConv = ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg)
    def forward(self, x):
        """Forward function."""
        b, c, h, w = x.size()

        # Spatial Attention (psa)
        input_x = self.conv_v_right(x)  #b, c/2, h, w
        context_mask = self.conv_q_right(x) #b, 1, h, w
        context_mask = torch.softmax(context_mask.reshape(b, 1, h * w), dim=2) #b, 1, h*w
        context_mask = context_mask.reshape(b, 1, h * w)
        context = torch.matmul(input_x.reshape(b, self.channels, h * w),
                               context_mask.transpose(1, 2)) #b, C/2, 1
        # spatial_attn = self.conv_up(F.layer_norm(context.reshape(b, c//2, 1, 1), normalized_shape=(c//2, 1, 1))).reshape(b, c, 1)   #b, c, 1
        spatial_attn = torch.sigmoid(F.layer_norm(context.reshape(b, self.channels, 1, 1), normalized_shape=(self.channels, 1, 1)))   #b, c/2, 1
        spatial_attn = self.conv_up(spatial_attn).reshape(b, c, 1) #b, c, 1
        # spatial_out = x * spatial_attn

        # Channel Attention (psa)
        g_x = self.conv_q_left(x)   #b, c/2, h, w
        avg_x = torch.softmax(F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, self.in_channels), dim=1) #b, c/2
        # avg_x = F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, self.channels)  # b, c/2
        avg_x = avg_x.reshape(b, 1, self.in_channels)
        theta_x = self.conv_v_left(x).reshape(b, self.in_channels, h * w)  #b, c/2, h*w
        context = torch.matmul(avg_x, theta_x)  #b, 1, h*w
        channel_attn = torch.sigmoid(context) #b, 1, h*w

        # channel_out = x * channel_attn

        sequence = torch.bmm(spatial_attn, channel_attn).reshape(b, c, h, w)

        if self.with_self is True:
            self_attn = torch.softmax(sequence.reshape(b, c, h * w), dim=2).reshape(b, c, h, w)
            value = self.resConv(x)
            self_attn = self_attn * value
            self_attn = self_attn.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
            out = self_attn
        else:
            out = x * sequence
        return out + x

@MODELS.register_module()
class EX_Module_noself(nn.Module):
    def __init__(self,
                 in_channels,
                 with_self=True,
                 # channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(EX_Module_noself, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 2
        self.with_self = with_self
        self.conv_q_right = ConvModule(in_channels,
                                       1,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_v_right = ConvModule(in_channels,
                                       self.channels,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_up = ConvModule(self.channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False,
                                  act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0),
                                  conv_cfg=conv_cfg)

        self.conv_q_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #g
        self.conv_v_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #theta
        # self.sk = SK_Module(in_channels=in_channels,
        #                     conv_cfg=conv_cfg)
        if with_self is True:
            self.resConv = ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg)
    def forward(self, x):
        """Forward function."""
        b, c, h, w = x.size()

        # Spatial Attention (psa)
        input_x = self.conv_v_right(x)  #b, c/2, h, w
        context_mask = self.conv_q_right(x) #b, 1, h, w
        context_mask = torch.softmax(context_mask.reshape(b, 1, h * w), dim=2) #b, 1, h*w
        context_mask = context_mask.reshape(b, 1, h * w)
        context = torch.matmul(input_x.reshape(b, c // 2, h * w),
                               context_mask.transpose(1, 2)) #b, C/2, 1
        # spatial_attn = self.conv_up(F.layer_norm(context.reshape(b, c//2, 1, 1), normalized_shape=(c//2, 1, 1))).reshape(b, c, 1)   #b, c, 1
        spatial_attn = torch.sigmoid(F.layer_norm(context.reshape(b, c//2, 1, 1), normalized_shape=(c//2, 1, 1)).reshape(b, c // 2, 1))   #b, c/2, 1
        # spatial_out = x * spatial_attn

        # Channel Attention (psa)
        g_x = self.conv_q_left(x)   #b, c/2, h, w
        avg_x = torch.softmax(F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, c // 2), dim=1) #b, c/2
        avg_x = avg_x.reshape(b, 1, c // 2)
        theta_x = self.conv_v_left(x).reshape(b, c // 2, h * w)  #b, c/2, h*w
        context = torch.matmul(avg_x, theta_x)  #b, 1, h*w
        channel_attn = torch.sigmoid(context) #b, 1, h*w

        # channel_out = x * channel_attn

        sequence = torch.bmm(spatial_attn, channel_attn).reshape(b, c // 2, h, w)

        parallel = (channel_attn + spatial_attn).reshape(b, c // 2, h, w)
        # sequence = (x * channel_attn) * spatial_attn
        #
        # parallel = x * channel_attn + x * spatial_attn
        # sk_attn = self.sk(sequence, parallel)
        sk_attn = torch.cat((sequence, parallel), dim=1)

        if self.with_self is True:
            self_attn = torch.softmax(sk_attn.reshape(b, c, h * w), dim=2).reshape(b, c, h, w)
            value = self.resConv(x)
            self_attn = self_attn * value
            self_attn = self_attn.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1)
            out = self_attn
        else:
            out = x * sk_attn
        return out + x

@MODELS.register_module()
class EX_Module_3D(nn.Module):
    def __init__(self,
                 in_channels,
                 with_self=True,
                 # channels,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(EX_Module_3D, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 2
        self.with_self = with_self
        self.conv_q_right = ConvModule(in_channels,
                                       1,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_v_right = ConvModule(in_channels,
                                       self.channels,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False,
                                       conv_cfg=conv_cfg)
        self.conv_up = ConvModule(self.channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False,
                                  act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0),
                                  conv_cfg=conv_cfg)

        self.conv_q_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #g
        self.conv_v_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      conv_cfg=conv_cfg)   #theta
        self.sk = SK_Module(in_channels=in_channels,
                            conv_cfg=conv_cfg)
        if with_self is True:
            self.resConv = ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg)
    def forward(self, x):
        """Forward function."""
        b, c, d, h, w = x.size()

        # Spatial Attention (psa)
        input_x = self.conv_v_right(x)  #b, c/2, d, h, w
        context_mask = self.conv_q_right(x) #b, 1, d, h, w
        context_mask = F.softmax(context_mask.reshape(b, 1, d * h * w), dim=2) #b, 1, d*h*w
        context_mask = context_mask.reshape(b, 1, d * h * w)
        context = torch.matmul(input_x.reshape(b, c // 2, d * h * w),
                               context_mask.transpose(1, 2)) #b, C/2, 1
        spatial_attn = self.conv_up(F.layer_norm(context.reshape(b, c//2, 1, 1, 1), normalized_shape=(c//2, 1, 1, 1))).reshape(b, c, 1)   #b, c, 1
        # spatial_out = x * spatial_attn

        # Channel Attention (psa)
        g_x = self.conv_q_left(x)   #b, c/2, d, h, w
        avg_x = F.softmax(F.adaptive_avg_pool3d(g_x, output_size=1).reshape(b, c // 2), dim=1) #b, c/2
        avg_x = avg_x.reshape(b, 1, c // 2)
        theta_x = self.conv_v_left(x).reshape(b, c // 2, d * h * w)  #b, c/2, d*h*w
        context = torch.matmul(avg_x, theta_x)  #b, 1, d*h*w
        channel_attn = torch.sigmoid(context) #b, 1, d*h*w

        # channel_out = x * channel_attn

        sequence = torch.bmm(spatial_attn, channel_attn).reshape(b, c, d, h, w)

        parallel = F.softmax((channel_attn + spatial_attn).reshape(b, c, d * h * w), dim=2).reshape(b, c, d, h, w)
        # sequence = (x * channel_attn) * spatial_attn
        #
        # parallel = x * channel_attn + x * spatial_attn
        sk_attn = self.sk(sequence, parallel)
        # sk_attn = torch.cat((sequence, parallel), dim=1)

        if self.with_self is True:
            self_attn = sk_attn.reshape(b, c, d * h * w).sum(dim=2).reshape(b, c, 1, 1, 1) + self.resConv(x)
            out = self_attn
        else:
            out = x * sk_attn
        return out

class EX_Module1(nn.Module):
    def __init__(self,
                 in_channels,
                 # channels,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(EX_Module1, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck1 = ConvModule(
            in_channels,
            in_channels // 2,
            kernel_size=1,
            act_cfg=dict(type='ReLU'))
        self.bottleneck2 = ConvModule(
            in_channels // 2,
            in_channels,
            kernel_size=1,
            # norm_cfg=dict(type='LN', normalized_shape=(CWH), eps=1e-6),
            act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0))
        # self.conv0 = ConvModule(
        #     in_channels,
        #     in_channels // 2,
        #     kernel_size=1,
        #     bias=False)
        self.conv1 = ConvModule(
            in_channels,
            1,
            kernel_size=1,
            bias=False,
            act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0))
        # self.conv2 = Conv2d(
        #     in_channels,
        #     channels,
        #     kernel_size=3,
        #     padding=1)
        self.sk_module = SK_Module(in_channels=in_channels)
        self.resConv = Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            bias=False)
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        # self.gmp = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def channel_attention(self, x):
        # channel attention
        b, c, h, w = x.size()
        # channel_attn = self.conv0(x)  # c/2,h,w
        channel_attn = self.gap(x)  # c,1,1
        channel_attn = self.bottleneck1(channel_attn)  # c/2, 1, 1
        channel_attn = self.bottleneck2(channel_attn)  # c, 1, 1
        # channel_attn = self.sigmoid(channel_attn)
        channel_attn = nchw_to_nlc(channel_attn)
        channel_attn = self.norm(channel_attn)
        channel_attn = channel_attn.reshape(b, c, 1, 1)

        return channel_attn

    def sequence_attention(self, spatial_attn, channel_attn):
        #sequence attention:a*c*s
        b, c, h, w = self.size
        spatial_attn = spatial_attn.reshape(b, 1, h * w)  # 1,h*w
        channel_attn = channel_attn.reshape(b, c, 1)  # c,1
        seq_attn = torch.bmm(channel_attn, spatial_attn)  # c,h,w
        # seq_attn = self.sigmoid(torch.bmm(channel_attn, spatial_attn))  # c,h,w
        seq_attn = seq_attn.reshape(b, c, h, w)
        # seq_attn = self.sigmoid(x * seq_attn)

        return seq_attn

    def selected_attention(self, x, seq_attn, par_attn, hw_shape):
        # select attention
        sk_results = self.sk_module(seq_attn, par_attn)
        # sk_results = nchw_to_nlc(sk_results)
        # sk_results = nn.Softmax(dim=1)(sk_results)
        # sk_results = nlc_to_nchw(sk_results, hw_shape)
        # selected_attn = x * sk_results  # c,h,w

        return sk_results

    def self_attention(self, x, selected_attn):
        # self attention
        b, c, h, w = self.size
        self_attn = selected_attn.reshape(b, c, h * w)  # b*c*n
        # self_attn = self.gap(selected_attn).reshape(b, c)
        # self_attn = nn.Softmax(dim=2)(self_attn) # b*c*1*1
        # self_attn = self_attn.reshape(b, c, h, w)  # b*c*n
        self_attn = self_attn.sum(dim=2).reshape(b, c, 1, 1)  # b*c*1*1

        self_attn_res = self.resConv(x)

        self_attn = self_attn + self_attn_res

        return self_attn

    def forward(self, x):
        """Forward function."""
        self.size = x.size()
        b, c, h, w = x.size()
        hw_shape = (h, w)

        # channel attention
        channel_attn = self.channel_attention(x)

        # spatial attention
        spatial_attn = self.conv1(x)  # 1, h, w
        # softmax spatial_attn
        # spatial_attn = spatial_attn.reshape(b, 1, h * w)
        # spatial_attn = nn.Softmax(dim=1)(spatial_attn)
        # spatial_attn = spatial_attn.reshape(b, 1, h, w)

        # parallel attention:a*c+a*s
        # par_attn = self.sigmoid(spatial_attn + channel_attn)
        par_attn = (spatial_attn + channel_attn) // 2
        # par_attn = self.sigmoid(x * par_attn)

        # sequence attention:a*c*s
        seq_attn = self.sequence_attention(spatial_attn, channel_attn)

        # select attention
        selected_attn = self.selected_attention(x, seq_attn, par_attn, hw_shape)
        # selected_attn = selected_attn.reshape(b, c, h * w)
        # selected_attn = nn.Softmax(dim=2)(selected_attn)
        # selected_attn = selected_attn.reshape(b, c, h, w)
        # self-attention
        # self_attn = self.softmax(sk_results)
        # self_attn = x * self_attn  # c,h,w
        self_attn = self.self_attention(x, selected_attn)

        # add attentions
        out = x * selected_attn + self_attn
        # out = self_attn
        # out = self.conv2(out)
        return out

# @MODELS.register_module()
class EX_Module2(nn.Module):
    def __init__(self,
                 in_channels,
                 with_self=True,
                 # channels,
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(EX_Module2, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 2
        self.with_self = with_self
        self.conv_q_right = ConvModule(in_channels,
                                       1,
                                       kernel_size=1,
                                       stride=1,
                                       bias=False)
        # self.conv_v_right = ConvModule(in_channels,
        #                                self.channels,
        #                                kernel_size=1,
        #                                stride=1,
        #                                bias=False)
        self.conv_up = ConvModule(self.channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False,
                                  act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0))

        self.conv_q_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False)   #g
        self.conv_v_left = ConvModule(in_channels,
                                      self.channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False)   #theta
        # self.sk = SK_Module(in_channels=in_channels)
        if with_self is True:
            self.resConv = ConvModule(
                in_channels,
                in_channels,
                kernel_size=1,
                bias=False)
    def forward(self, x):
        """Forward function."""
        b, c, h, w = x.size()
        hw_shape = (h, w)

        # Spatial Attention (psa)
        # input_x = self.conv_v_right(x)
        context_mask = self.conv_q_right(x) #b, 1, h, w
        context_mask = F.softmax(context_mask.reshape(b, 1, h * w), dim=2) #b, 1, h*w
        spatial_attn = context_mask.reshape(b, 1, h * w)
        # context = torch.matmul(input_x.reshape(b, c // 2, h * w), context_mask.transpose(1, 2)).unsqueeze(-1) #b, C/2, 1, 1
        # spatial_attn = self.conv_up(F.layer_norm(context, normalized_shape=(c//2, 1, 1)))
        # spatial_out = x * spatial_attn

        # Channel Attention (psa)
        g_x = self.conv_q_left(x)   #b, c/2, h, w
        avg_x = F.softmax(F.adaptive_avg_pool2d(g_x, output_size=1).reshape(b, c // 2), dim=1) #b, c/2
        channel_attn = avg_x.reshape(b, c // 2, 1)
        # channel_attn = self.conv_up(channel_attn).reshape(b, c, 1)

        # softmax_x = self.conv_v_left(x)  #b, c/2, h, w
        # channel_attn = HSigmoid(context)
        # channel_out = x * channel_attn

        sequence = torch.bmm(channel_attn, spatial_attn).reshape(b, c // 2, h, w)

        parallel = F.softmax((channel_attn + spatial_attn).reshape(b, c // 2, h * w), dim=2).reshape(b, c // 2, h, w)
        # sequence = (x * channel_attn) * spatial_attn
        #
        # parallel = x * channel_attn + x * spatial_attn
        # sk_attn = self.sk(sequence, parallel)
        sk_attn = torch.cat((sequence, parallel), dim=1)

        if self.with_self is True:
            self_attn = sk_attn.reshape(b, c, h * w).sum(dim=2).reshape(b, c, 1, 1) + self.resConv(x)
            out = x * sk_attn + self_attn
        else:
            out = x * sk_attn
        return out

class EX_Module_noselect_seq(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 norm_cfg=None):
        super(EX_Module_noselect_seq, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck1 = Conv2d(
            in_channels // 2,
            in_channels,
            kernel_size=1)
        self.conv0 = Conv2d(
            in_channels,
            in_channels // 2,
            kernel_size=1)
        self.conv1 = Conv2d(
            in_channels,
            1,
            kernel_size=1)
        self.sigmoid = HSigmoid()
        # self.conv2 = Conv2d(
        #     in_channels,
        #     channels,
        #     kernel_size=3,
        #     padding=1)
        # self.sk_module = SK_Module(in_channels=in_channels)
        self.resConv = Conv2d(
            in_channels,
            in_channels,
            kernel_size=1)
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, in_channels)[1]

    def channel_attention(self, x):
        # channel attention
        b, c, h, w = x.size()
        channel_attn = self.conv0(x)  # c/2,h,w
        channel_attn = self.gap(channel_attn)  # c/2,1,1
        channel_attn = self.bottleneck1(channel_attn)  # c, 1, 1
        channel_attn = self.sigmoid(channel_attn)
        channel_attn = nchw_to_nlc(channel_attn)
        channel_attn = self.norm(channel_attn)
        channel_attn = channel_attn.reshape(b, c, 1, 1)

        return channel_attn

    def sequence_attention(self, spatial_attn, channel_attn):
        #sequence attention:a*c*s
        b, c, h, w = self.size
        spatial_attn = spatial_attn.reshape(b, 1, h * w)  # 1,h*w
        channel_attn = channel_attn.reshape(b, c, 1)  # c,1
        seq_attn = self.sigmoid(torch.bmm(channel_attn, spatial_attn))  # c,h,w
        seq_attn = seq_attn.reshape(b, c, h, w)
        # seq_attn = self.sigmoid(x * seq_attn)

        return seq_attn

    def self_attention(self, x, selected_attn):
        # self attention
        b, c, h, w = self.size
        self_attn = selected_attn.reshape(b, c, h * w)  # b*c*n
        self_attn = self_attn.sum(dim=2).reshape(b, c, 1, 1)  # b*c*1*1

        self_attn_res = self.resConv(x)

        self_attn = self_attn + self_attn_res

        return self_attn

    def forward(self, x):
        """Forward function."""
        self.size = x.size()
        b, c, h, w = x.size()
        hw_shape = (h, w)

        # channel attention
        channel_attn = self.channel_attention(x)

        # spatial attention
        spatial_attn = self.conv1(x)  # 1, h, w

        # sequence attention:a*c*s
        seq_attn = self.sequence_attention(spatial_attn, channel_attn)

        # self-attention
        # self_attn = self.softmax(sk_results)
        # self_attn = x * self_attn  # c,h,w
        self_attn = self.self_attention(x, seq_attn)

        # add attention
        out = seq_attn + self_attn

        # out = self.conv2(out)
        return out

class EX_Module_noselect_par(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 norm_cfg=None):
        super(EX_Module_noselect_par, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck1 = Conv2d(
            in_channels // 2,
            in_channels,
            kernel_size=1)
        self.conv0 = Conv2d(
            in_channels,
            in_channels // 2,
            kernel_size=1)
        self.conv1 = Conv2d(
            in_channels,
            1,
            kernel_size=1)
        self.sigmoid = HSigmoid()
        # self.conv2 = Conv2d(
        #     in_channels,
        #     channels,
        #     kernel_size=3,
        #     padding=1)
        # self.sk_module = SK_Module(in_channels=in_channels)
        self.resConv = Conv2d(
            in_channels,
            in_channels,
            kernel_size=1)
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, in_channels)[1]

    def channel_attention(self, x):
        # channel attention
        b, c, h, w = x.size()
        channel_attn = self.conv0(x)  # c/2,h,w
        channel_attn = self.gap(channel_attn)  # c/2,1,1
        channel_attn = self.bottleneck1(channel_attn)  # c, 1, 1
        channel_attn = self.sigmoid(channel_attn)
        channel_attn = nchw_to_nlc(channel_attn)
        channel_attn = self.norm(channel_attn)
        channel_attn = channel_attn.reshape(b, c, 1, 1)

        return channel_attn

    def self_attention(self, x, selected_attn):
        # self attention
        b, c, h, w = self.size
        self_attn = selected_attn.reshape(b, c, h * w)  # b*c*n
        self_attn = self_attn.sum(dim=2).reshape(b, c, 1, 1)  # b*c*1*1

        self_attn_res = self.resConv(x)

        self_attn = self_attn + self_attn_res

        return self_attn

    def forward(self, x):
        """Forward function."""
        self.size = x.size()
        b, c, h, w = x.size()
        hw_shape = (h, w)

        # channel attention
        channel_attn = self.channel_attention(x)

        # spatial attention
        spatial_attn = self.conv1(x)  # 1, h, w

        # parallel attention:a*c+a*s
        par_attn = self.sigmoid(spatial_attn + channel_attn)
        # par_attn = self.sigmoid(x * par_attn)

        # self-attention
        # self_attn = self.softmax(sk_results)
        # self_attn = x * self_attn  # c,h,w
        self_attn = self.self_attention(x, par_attn)

        # add attention
        out = par_attn + self_attn

        # out = self.conv2(out)
        return out

class EX_Module_onlyselect(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 norm_cfg=None):
        super(EX_Module_onlyselect, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck1 = Conv2d(
            in_channels // 2,
            in_channels,
            kernel_size=1)
        self.conv0 = Conv2d(
            in_channels,
            in_channels // 2,
            kernel_size=1)
        self.conv1 = Conv2d(
            in_channels,
            1,
            kernel_size=1)
        self.sigmoid = HSigmoid()
        # self.conv2 = Conv2d(
        #     in_channels,
        #     channels,
        #     kernel_size=3,
        #     padding=1)
        self.sk_module = SK_Module(in_channels=in_channels)
        self.resConv = Conv2d(
            in_channels,
            in_channels,
            kernel_size=1)
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, in_channels)[1]

    def channel_attention(self, x):
        # channel attention
        b, c, h, w = x.size()
        channel_attn = self.conv0(x)  # c/2,h,w
        channel_attn = self.gap(channel_attn)  # c/2,1,1
        channel_attn = self.bottleneck1(channel_attn)  # c, 1, 1
        channel_attn = self.sigmoid(channel_attn)
        channel_attn = nchw_to_nlc(channel_attn)
        channel_attn = self.norm(channel_attn)
        channel_attn = channel_attn.reshape(b, c, 1, 1)

        return channel_attn

    def sequence_attention(self, spatial_attn, channel_attn):
        #sequence attention:a*c*s
        b, c, h, w = self.size
        spatial_attn = spatial_attn.reshape(b, 1, h * w)  # 1,h*w
        channel_attn = channel_attn.reshape(b, c, 1)  # c,1
        seq_attn = self.sigmoid(torch.bmm(channel_attn, spatial_attn))  # c,h,w
        seq_attn = seq_attn.reshape(b, c, h, w)
        # seq_attn = self.sigmoid(x * seq_attn)

        return seq_attn

    def selected_attention(self, x, seq_attn, par_attn, hw_shape):
        # select attention
        sk_results = self.sk_module(seq_attn, par_attn)
        sk_results = nchw_to_nlc(sk_results)
        sk_results = nn.Softmax(dim=1)(sk_results)
        sk_results = nlc_to_nchw(sk_results, hw_shape)
        selected_attn = x * sk_results  # c,h,w

        return selected_attn

    def forward(self, x):
        """Forward function."""
        self.size = x.size()
        b, c, h, w = x.size()
        hw_shape = (h, w)

        # channel attention
        channel_attn = self.channel_attention(x)

        # spatial attention
        spatial_attn = self.conv1(x)  # 1, h, w

        # parallel attention:a*c+a*s
        par_attn = self.sigmoid(spatial_attn + channel_attn)
        # par_attn = self.sigmoid(x * par_attn)

        # sequence attention:a*c*s
        seq_attn = self.sequence_attention(spatial_attn, channel_attn)

        # select attention
        selected_attn = self.selected_attention(x, seq_attn, par_attn, hw_shape)

        # add attention
        out = selected_attn + x

        # out = self.conv2(out)
        return out