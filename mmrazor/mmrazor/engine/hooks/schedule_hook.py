# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from mmseg.registry import HOOKS
from mmengine.logging import MMLogger, print_log
@HOOKS.register_module()
class DistillScheduleHook(Hook):
    def __init__(self,
                 interval1,
                 interval2):
        self.interval1 = interval1
        self.interval2 = interval2

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        # if runner.iter = self.interval:
        #     new_plugins = [dict(cfg=dict(type='EX_Module', with_self=False),
        #                                           stages=(True, True, True, True),
        #                                           position='after_conv1')]
        #     runner.model.backbone.plugins = new_plugins
        #     logger: MMLogger = MMLogger.get_current_instance()
        #     print_log(f'plugins change to {new_plugins}', logger)
        if runner.iter >= self.interval1 and runner.iter <= self.interval2:
            runner.model.distiller.distill_losses.loss_mgd.a = 2.0 - runner.iter / self.interval1
            # x=(runner.iter-(self.interval1+self.interval2)/2.0) / ((self.interval2-self.interval1)/20.0)
            # runner.model.distiller.distill_losses.loss_mgd.a = 1 / (1+np.exp(x))
            # runner.model.module.distill_losses.loss_mgd_fea.flag1 = False
        # elif runner.iter == self.interval2:
        #     # runner.model.module.distill_losses.loss_mgd_fea.flag2 = False
        #     runner.model.distiller.distill_losses.loss_mgd.a = 0



