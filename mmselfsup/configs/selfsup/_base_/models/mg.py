_base_ = [
    '../datasets/synapse_mg.py',
    '../schedules/sgd_steplr-200e_in1k.py',
    '../default_runtime.py',
]

# model settings
model = dict(
    type='ModelGenesis',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='UNet2D',
        n_class=3),
    head=dict(
        type='MGHead',
        n_class=1,
        loss=dict(type='mmcls.CrossEntropyLoss'),
    )
)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50,
))

optimizer = dict(type='SGD', lr=1, weight_decay=0, momentum=0.9)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# learning rate scheduler

param_scheduler = [dict(type='StepLR', step_size=10, by_epoch=True, gamma=0.5)]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse-selfsup',
            name='mg-pretrain'))
]
visualizer = dict(
    type='SelfSupVisualizer', vis_backends=vis_backends, name='visualizer')