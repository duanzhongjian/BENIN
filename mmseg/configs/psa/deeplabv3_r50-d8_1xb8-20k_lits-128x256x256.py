_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/lits_64x256x256.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (64, 256, 256)
data_preprocessor = dict(
    _scope_='mmseg',
    type='SegDataPreProcessor',
    mean=-111.58,#-105,-111.58
    std=119.48,#124,119.48
    # mean=None,  # -105,-111.58
    # std=None,  # 124,119.48
    # bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0,
    size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(in_channels=64),
    decode_head=dict(num_classes=2,
                     out_channels=2,
                     out_channel3d=64,
                     loss_decode=[
                         dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
                         dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]),
    auxiliary_head=dict(out_channel3d=128,
                        num_classes=2),
    test_cfg=dict(mode='slide_3d', crop_size=(64, 256, 256), stride=(64, 170, 170))
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=2000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=2, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw_ct=True, interval=10),
    checkpoint=dict(type='CheckpointHook',
                      by_epoch=False,
                      interval=2000,
                      max_keep_ckpts=3,
                      save_best=['mDice'], rule='greater'))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='LITS-3', name='resnet-20k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')