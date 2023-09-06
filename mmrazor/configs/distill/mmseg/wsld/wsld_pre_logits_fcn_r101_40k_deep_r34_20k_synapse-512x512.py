_base_ = [
    'mmseg::_base_/datasets/synapse.py',
    'mmseg::_base_/schedules/schedule_20k.py',
    'mmseg::_base_/default_runtime.py'
]

teacher_ckpt = './checkpoints/72.65best_mIoU_iter_40000.pth'  # noqa: E501
teacher_cfg_path = 'mmseg::psa/fcn_pre-r101-d8_1xb2-40k_synapse-512x512.py'  # noqa: E501
student_cfg_path = 'mmseg::psa/deeplabv3_r34-d8_1xb8-20k_synapse-512x512.py'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    # data_preprocessor=dict(
    #     type='ImgDataPreprocessor',
    #     # RGB format normalization parameters
    #     mean=[123.675, 116.28, 103.53],
    #     std=[58.395, 57.12, 57.375],
    #     # convert image from BGR to RGB
    #     bgr_to_rgb=True),
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.conv_seg'),
            gt_labels=dict(type='ModuleInputs', source='decode_head.loss_decode')),
        teacher_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.conv_seg')),
        distill_losses=dict(
            loss_wsld=dict(type='WSLD', tau=1, loss_weight=2.5, num_classes=14)),
        loss_forward_mappings=dict(
            loss_wsld=dict(
                student=dict(recorder='logits', from_student=True),
                teacher=dict(recorder='logits', from_student=False),
                gt_labels=dict(
                    recorder='gt_labels', from_student=True, data_idx=1)))))

find_unused_parameters = True

optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

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

# val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse',
            name='wsld-pre-fcn-r101-deep-r34-20k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    _scope_='mmseg',
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
