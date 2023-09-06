# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from mmseg.registry import HOOKS
from mmengine.logging import MMLogger, print_log
@HOOKS.register_module()
class TrainingScheduleHook(Hook):
    def __init__(self,
                 interval,
                 use_fcn=False):
        self.interval = interval
        self.use_fcn = use_fcn

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:

        if runner.iter == self.interval:
            new_plugins = [dict(cfg=dict(type='EX_Module', with_self=True),
                                                  stages=(True, True, True, True),
                                                  position='after_conv1')]
            runner.model.backbone.plugins = new_plugins
            logger: MMLogger = MMLogger.get_current_instance()
            if self.use_fcn:
                runner.model.decode_head.with_self = True
                print_log(f'decode_head.with_self change to {True}', logger)
            print_log(f'plugins change to {new_plugins}', logger)



