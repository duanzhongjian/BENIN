_base_ = [
    'mmseg::_base_/datasets/chestxray_512x512.py',
    'mmseg::_base_/schedules/schedule_20k.py',
    'mmseg::_base_/default_runtime.py'
]

teacher_ckpt = './work_dirs/deeplabv3_r50-d8_1xb2-40k_chestxray-512x512/iter_40000.pth' # noqa: E501
teacher_cfg_path = 'mmseg::deeplabv3/deeplabv3_r50-d8_1xb2-40k_chestxray-512x512.py'  # noqa: E501
student_cfg_path = 'mmseg::deeplabv3/deeplabv3_r18-d8_1xb2-20k_chestxray-512x512.py'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        distill_losses=dict(
            loss_cwd=dict(type='ChannelWiseDivergence', tau=1, loss_weight=5)),
        student_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.conv_seg')),
        teacher_recorders=dict(
            logits=dict(type='ModuleOutputs', source='decode_head.conv_seg')),
        loss_forward_mappings=dict(
            loss_cwd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
