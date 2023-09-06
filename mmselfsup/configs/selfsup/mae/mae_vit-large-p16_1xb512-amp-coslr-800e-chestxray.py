_base_ = 'mae_vit-large-p16_8xb512-amp-coslr-800e_in1k.py'

# dataset
dataset_type = 'mmcls.ImageNet'
data_root = 'data/images/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RandomResizedCrop',
        size=1024,
        scale=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='training/'),
        pipeline=train_pipeline))

# optimizer
optimizer = dict(
    lr=1.5e-4 * 4096 / 256 * (2 / 512 * 8),
)
