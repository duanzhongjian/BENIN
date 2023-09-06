_base_ = [
    '../_base_/models/deeplabv3_r50_3d-d8.py', '../_base_/datasets/lits_tumor_64x256x256.py',
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
    backbone=dict(in_channels=1,
                  maxpool_dim='3d'),
    decode_head=dict(num_classes=3,
                     out_channels=3,
                     out_channel3d=64),
    test_cfg=dict(mode='slide_3d', crop_size=(64, 256, 256), stride=(64, 170, 170))
)


param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        begin=0,
        end=20000,
        by_epoch=False,
    )
]

optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

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
            project='LITS-3', name='r50_3d-20k-tumor'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')