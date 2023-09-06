_base_ = [
    'mmseg::_base_/datasets/chestxray_512x512.py',
    'mmseg::_base_/schedules/schedule_20k.py',
    'mmseg::_base_/default_runtime.py'
]

teacher_ckpt = './work_dirs/deeplabv3_r50-d8_1xb2-40k_chestxray-512x512/iter_40000.pth'  # noqa: E501
teacher_cfg_path = 'mmseg::psa/deeplabv3_r50_3d-d8_1xb8-40k_lits_tumor-128x256x256.py'  # noqa: E501
student_cfg_path = 'mmseg::psa/deeplabv3_r18_3d-d8_1xb8-20k_lits_tumor-128x256x256.py'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        distill_losses=dict(
            loss_cwd=dict(type='ChannelWiseDivergence', tau=1, loss_weight=3),
            loss_mgd=dict(type='FeatureLoss',
                          student_channels=512,
                          teacher_channels=2048,
                          alpha_mgd=0.00002,
                          lambda_mgd=0.75)),
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

# val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw_table=True, interval=10))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='chest X-rays',
            name='mgd-r50-r18-20k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(
    _scope_='mmseg',
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

