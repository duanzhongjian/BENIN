_base_ = 'mae_vit-base-p16_1xb2-coslr-400e_synapse.py'

# mixed precision
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')

# pre-train for 100 epochs
train_cfg = dict(max_epochs=400)

# model settings
model = dict(
    backbone=dict(type='MAEViT', arch='h', patch_size=14, mask_ratio=0.125),
    neck=dict(
        type='MAEPretrainDecoder',
        embed_dim=1280,
        patch_size=14,
        num_patches=256),
    head=dict(patch_size=14))

# learning rate scheduler
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=400,
        by_epoch=True,
    )
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse',
            name='mae-400e'))
]
visualizer = dict(
    type='SelfSupVisualizer', vis_backends=vis_backends, name='visualizer')
