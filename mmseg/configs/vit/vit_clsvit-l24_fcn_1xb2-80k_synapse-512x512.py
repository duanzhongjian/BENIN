_base_ = [
    '../_base_/models/fcn_clsvit-b16.py',
    '../_base_/datasets/synapse.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained',
                      checkpoint='pretrain/vit-large395.pth'),
        # frozen_stages=24,
        arch='l',
        out_indices=(5, 11, 17, 23)
    ),
    decode_head=dict(in_channels=1024,
                     num_classes=14),
    auxiliary_head=dict(in_channels=1024,
                        num_classes=14))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# learning rate scheduler
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=80000,
        by_epoch=False,
    )
]
# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw_table=True, interval=50),
    checkpoint=dict(type='CheckpointHook',
                    by_epoch=False,
                    interval=8000,
                    max_keep_ckpts=3,
                    save_best=['mIoU'], rule='greater'))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse',
            name='clsvit-l24-fcn-80k'),
        define_metric_cfg=dict(mIou='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

