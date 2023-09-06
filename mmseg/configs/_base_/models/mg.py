_base_ = [
    '../datasets/imagenet_mg.py',
    '../schedules/schedule_40k.py',
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
        n_class=13),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=13,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=13,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    )
)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50,
))

# optimizer = dict(type='SGD', lr=1, weight_decay=0, momentum=0.9)
# optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# learning rate scheduler
# param_scheduler = [dict(type='StepLR', step_size=10, by_epoch=True, gamma=0.5)]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse-selfsup',
            name='mg-pretrained'))
]
visualizer = dict(
    type='SelfSupVisualizer', vis_backends=vis_backends, name='visualizer')