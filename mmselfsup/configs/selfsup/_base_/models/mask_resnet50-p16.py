# model settings
model = dict(
    type='MaskRes',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='MaskResNetV1c',
        depth=101,
        in_channels=3,
        out_indices=[5],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    head=dict(
        type='MGHead',
        n_class=3,
        loss=dict(type='mmcls.CrossEntropyLoss')))
