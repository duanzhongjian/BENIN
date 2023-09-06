_base_ = 'mae_vit-base-p16_1xb2-coslr-400e_synapse.py'

# mixed precision
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')

# pre-train for 100 epochs
train_cfg = dict(max_epochs=1600)

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
        end=1600,
        by_epoch=True,
    )
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse',
            name='mae-1600e'))
]
visualizer = dict(
    type='SelfSupVisualizer', vis_backends=vis_backends, name='visualizer')
