_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/lits17.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

# class_weight = [0.00445, 0.51355, 1.58497, 1.05022, 4.07077, 14.30097, 0.29402, 0.31209, 0.69898, 0.26710, 1.00000, 2.60781, 3.48121, 1.34730]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    # visualize=True,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    decode_head=dict(num_classes=3,
                     # loss_decode=dict(
                     #     type='CrossEntropyLoss', class_weight=class_weight)
                     ),
    auxiliary_head=dict(num_classes=3,
                        # loss_decode=dict(
                        #     type='CrossEntropyLoss', class_weight=class_weight)
                        ),
    test_cfg=dict(mode='whole')
)

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(
#         type='PolyLR',
#         eta_min=1e-3,
#         power=2.0,
#         begin=1500,
#         end=20000,
#         by_epoch=False,
#     )
# ]
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=20000,
        by_epoch=False,
    )
]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

train_dataloader = dict(batch_size=2, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw_table=True, interval=50),
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
            project='lits-17', name='deep-r50-20k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')