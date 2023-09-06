_base_ = ['./deeplabv3_r50-d8_4xb2-20k_lits-32x256x256.py']

# model = dict(backbone=dict(with_attn='ex', with_attn_stage=[2, 2, 2, 2]))

crop_size = (128, 256, 256)
data_preprocessor = dict(
    _scope_='mmseg',
    type='SegDataPreProcessor',
    mean=None,#-105,
    std=None,#124,
    # bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0,
    size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(in_channels=128,
                  plugins=[dict(cfg=dict(type='EX_Module'),
                                stages=(True, True, True, True),
                                position='after_conv3')]))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='LITS-3', name='resnet-ex-20k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')