_base_ = [
    '../_base_/models/mask_resnet50-p16.py',
    '../_base_/datasets/lits17.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# dataset 8 x 512
train_dataloader = dict(batch_size=4, num_workers=8)

# optimizer_wrapper
optimizer = dict(type='AdamW', lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.0005)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer, loss_scale='dynamic')

# # learning rate scheduler
# param_scheduler = [
#     dict(
#         type='PolyLR',
#         eta_min=1e-4,
#         power=0.9,
#         begin=0,
#         end=1600,
#         by_epoch=True,
#     )
# ]

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR', T_max=4960, by_epoch=True, begin=40, end=5000)
]

# runtime settings
# pre-train for 1000 epochs
train_cfg = dict(max_epochs=5000)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='lits-17',
            name='mask_res-5000e'))
]
visualizer = dict(
    type='SelfSupVisualizer', vis_backends=vis_backends, name='visualizer')
