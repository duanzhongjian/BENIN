_base_ = ['./deeplabv3_r18-d8_1xb8-20k_synapse-512x512.py']

model = dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type='CBAM'),
                      stages=(True, True, True, True),
                      position='after_conv3'),
                 ]
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='r18-cbam-20k'),
        define_metric_cfg=dict(mIoU='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')