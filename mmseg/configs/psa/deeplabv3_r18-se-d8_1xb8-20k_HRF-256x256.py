_base_ = ['./deeplabv3_r18-d8_1xb8-20k_HRF-256x256.py']

model = dict(
    backbone=dict(plugins=[dict(cfg=dict(type='SELayer'),
                                stages=(True, True, True, True),
                                position='after_conv1')]))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='HRF', name='r18-se-20k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')