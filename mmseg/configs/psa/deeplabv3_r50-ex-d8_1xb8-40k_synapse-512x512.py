_base_ = ['./deeplabv3_r50-d8_1xb8-40k_synapse-512x512.py']

model = dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type='EX_Module', with_self=False),
                      stages=(True, True, True, True),
                      position='after_conv1'),
                 ]
    )
)

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=2000),
#     dict(
#         type='PolyLR',
#         eta_min=1e-6,
#         power=0.9,
#         begin=2000,
#         end=40000,
#         by_epoch=False,
#     )
# ]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='r50-ex-40k'),
        define_metric_cfg=dict(mIoU='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

custom_hooks = [
    dict(type='TrainingScheduleHook', interval=4000)
]