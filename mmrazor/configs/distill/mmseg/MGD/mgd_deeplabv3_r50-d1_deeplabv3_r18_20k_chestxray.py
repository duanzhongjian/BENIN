_base_ = [
    'mmseg::_base_/datasets/chestxray_512x512.py',
    'mmseg::_base_/schedules/schedule_20k.py',
    'mmseg::_base_/default_runtime.py'
]

teacher_ckpt = './work_dirs/deeplabv3_r50-d8_1xb2-40k_chestxray-512x512/iter_40000.pth'
teacher_cfg_path = 'mmseg::deeplabv3/deeplabv3_r18-d8_1xb2-20k_chestxray-512x512.py'  # noqa: E501
student_cfg_path = 'mmseg::deeplabv3/deeplabv3_r50-d8_1xb2-40k_chestxray-512x512.py'  # noqa: E501

# model settings
find_unused_parameters=True
alpha_mgd=0.00002
lambda_mgd=0.75

model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='SegmentationDistiller',
        teacher_pretrained='',
        init_student=False,
        use_logit=True,
        distill_cfg=[dict(methods=[dict(type='FeatureLoss',
                                        name='loss_mgd_fea',
                                        student_channels=512,
                                        teacher_channels=2048,
                                        alpha_mgd=alpha_mgd,
                                        lambda_mgd=lambda_mgd,
                                        )
                                   ]
                          ),
                     ]
    )
)

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

