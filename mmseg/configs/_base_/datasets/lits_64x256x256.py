# dataset settings
dataset_type = 'LITSDataset'
data_root = 'data/LITS2'
# img_scale = (75, 256, 256)
crop_size = (64, 256, 256)
train_pipeline = [
    dict(type='LoadBiomedicalImageFromFile'),
    dict(type='LoadBiomedicalAnnotation'),
    dict(type='BioPatchCrop',
         crop_size=crop_size,
         crop_mode='nnUNet'),
    dict(type='MedPad', size=crop_size),
    # dict(type='MedicalRandomFlip',
    #      prob=[0.5, 0.3, 0.2],
    #      direction=[0, 1, 2]),
    # dict(type='ZNormalization', channel_wise=True),
    dict(type='PackMedicalInputs')
]
test_pipeline = [
    dict(type='LoadBiomedicalImageFromFile'),
    dict(type='LoadBiomedicalAnnotation'),
    dict(type='PackMedicalInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training',
            seg_map_path='annotations/training'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = val_evaluator
