_base_ = [
    'mmcls::_base_/models/swin_transformer/base_224.py',
    '../_base_/datasets/imagenet_swin_192.py',
    '../_base_/schedules/adamw_coslr-100e_in1k.py',
    'mmcls::_base_/default_runtime.py'
]
# SimMIM fine-tuning setting

# model settings
model = dict(
    backbone=dict(
        img_size=192,
        drop_path_rate=0.1,
        stage_cfgs=dict(block_cfgs=dict(window_size=6))))

# schedule settings
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=5e-3, model_type='swin', layer_decay_rate=0.9),
    clip_grad=dict(max_norm=5.0),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
    constructor='mmselfsup.LearningRateDecayOptimWrapperConstructor')

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=2.5e-7 / 1.25e-3,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=80,
        eta_min=2.5e-7 * 2048 / 512,
        by_epoch=True,
        begin=20,
        end=100,
        convert_to_iter_based=True)
]

# runtime settings
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=100))

randomness = dict(seed=0)
