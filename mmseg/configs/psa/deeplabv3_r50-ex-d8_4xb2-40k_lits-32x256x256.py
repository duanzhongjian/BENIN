_base_ = ['./deeplabv3_r50-d8_4xb2-40k_lits-32x256x256.py']

# model = dict(backbone=dict(with_attn='ex', with_attn_stage=[2, 2, 2, 2]))

model = dict(backbone=dict(
    plugins=[
        dict(
            cfg=dict(type='EX_Module'),
             stages=(True, True, True, True),
             position='after_conv3')]))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='LITS-3', name='resnet-ex-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')