# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel

__all__ = ['ReplaceDepthHook']


@HOOKS.register_module()
class ReplaceDepthHook(Hook):
    """ 
    example:

    dict(
        type='ReplaceDepthHook',
        replace_start_iter=300,
    ),
    """

    def __init__(self, replace_start_iter=0):
        super().__init__()
        self.replace_start_iter=replace_start_iter

    def set_replace_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            # view_transformation
            runner.model.module.module.use_depth_gt = flag
        else:
            # view_transformation
            runner.model.module.use_depth_gt = flag
    
    def before_iter(self, runner):
        if runner.iter > self.replace_start_iter:
            self.set_replace_flag(runner, True)

    def before_run(self, runner):
        #if runner.mode == 'train':
        self.set_replace_flag(runner, False)


