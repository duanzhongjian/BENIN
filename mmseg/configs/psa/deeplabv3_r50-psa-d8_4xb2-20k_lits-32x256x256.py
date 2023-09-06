_base_ = ['./deeplabv3_r50-d8_4xb2-20k_lits-32x256x256.py']

model = dict(backbone=dict(
    plugins=[
        dict(
            cfg=dict(type='PSA_p'),
             stages=(True, True, True, True),
             position='after_conv1')]))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='LITS-3', name='resnet-psa-20k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')