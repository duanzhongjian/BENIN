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
    _delete_=True,
    type='SingleTeacherDistill',
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.2.conv2')
            ),
        teacher_recorders=dict(
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.2.conv3')
            ),
        connectors=dict(
            bb_s4_connector=dict(
                type='MGDConnector',
                student_channels=512,
                teacher_channels=2048,
                lambda_mgd=0.75)
            ),
        distill_losses=dict(
            loss_mgd_bb_s4=dict(type='BVDKDLoss', alpha_mgd=0.00002)
        ),
        loss_forward_mappings=dict(
            loss_mgd_bb_s4=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='bb_s4'),
                preds_T=dict(from_student=False, recorder='bb_s4'),
                new_fea=dict(
                    from_student=True,
                    recorder='bb_s4',
                    connector='bb_s4_connector')),
        )))

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
                    save_best=['mDice'], rule='greater'))


# val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse',
            name='bvdkd-fcn-r101-deep-r34-20k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    _scope_='mmseg',
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

custom_hooks = [
    dict(type='DistillScheduleHook', interval1=5000, interval2=10000)
]
