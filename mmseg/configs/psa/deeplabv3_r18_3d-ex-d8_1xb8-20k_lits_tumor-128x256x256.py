_base_ = ['deeplabv3_r18_3d-d8_1xb8-20k_lits_tumor-128x256x256.py']

model = dict(
    backbone=dict(
        plugins=[dict(cfg=dict(type='EX_Module_3D', with_self=False),
                      stages=(True, True, True, True),
                      position='after_conv1'),
                 ]
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='LITS-3', name='r18_3d-ex-20k-tumor'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

custom_hooks = [
    dict(type='TrainingScheduleHook', interval=1000)
]