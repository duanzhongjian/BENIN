_base_ = [
    'mmseg::_base_/datasets/synapse.py',
    'mmseg::_base_/schedules/schedule_20k.py',
    'mmseg::_base_/default_runtime.py'
]

teacher_ckpt = './checkpoints/best_mIoU_iter_40000.pth'  # noqa: E501
teacher_cfg_path = 'mmseg::psa/fcn_r101-d8_1xb2-40k_synapse-512x512.py'  # noqa: E501
student_cfg_path = 'mmseg::psa/fcn_r50-d8_1xb8-20k_synapse-512x512.py'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        distill_losses=dict(
            loss_cwd=dict(type='ChannelWiseDivergence', tau=4, loss_weight=3),
            loss_mgd=dict(type='FeatureLoss',
                          student_channels=512,
                          teacher_channels=2048,
                          alpha_mgd=0.00002,
                          lambda_mgd=0.50)),
        student_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.conv_seg'),
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.1.conv2')),
        teacher_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.conv_seg'),
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.2.conv3')),
        loss_forward_mappings=dict(
            loss_cwd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits')),
            loss_mgd=dict(
                preds_S=dict(from_student=True, recorder='bb_s4'),
                preds_T=dict(from_student=False, recorder='bb_s4')))
    )
)

find_unused_parameters = True

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
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

train_dataloader = dict(batch_size=2, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw_table=True, interval=10),
    checkpoint=dict(type='CheckpointHook',
                    by_epoch=False,
                    interval=2000,
                    max_keep_ckpts=3,
                    save_best=['mIoU'], rule='greater'))


# val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse',
            name='mgd-fcn-r101-r50-20k'),
        define_metric_cfg=dict(mIoU='max'))
]
visualizer = dict(
    _scope_='mmseg',
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

custom_hooks = [
    dict(type='DistillScheduleHook', interval1=5000, interval2=10000)
]

