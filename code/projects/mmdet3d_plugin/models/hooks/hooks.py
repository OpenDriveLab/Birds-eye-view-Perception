from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time


@HOOKS.register_module()
class GradChecker(Hook):

    def after_train_iter(self, runner):
        for key, val in runner.model.named_parameters():
            if val.grad == None and val.requires_grad:
                print('WARNNING: {key}\'s parameters are not be used!!!!'.format(key=key))


@HOOKS.register_module()
class TransferWeight(Hook):

    def after_train_iter(self, runner):
        if runner.inner_iter < 500:  # warmup period
            runner.eval_model.load_state_dict(runner.model.state_dict())
        elif self.every_n_inner_iters(runner, 5):
            runner.eval_model.load_state_dict(runner.model.state_dict())


@HOOKS.register_module()
class MomentumUpdateWeight(Hook):
    def __init__(self, m=0.999) -> None:
        self.m = m
        super().__init__()

    # @run_time('MomentumUpdateWeight') about 0.02s once
    def after_train_iter(self, runner):
        for param_q, param_k in zip(runner.model.parameters(), runner.eval_model.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

# Copyright (c) OpenMMLab. All rights reserved.
import math

from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


class BaseEMAHook(Hook):
    """Exponential Moving Average Hook.

    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below. EMAHook takes priority over EvalHook and CheckpointHook. Note,
    the original model parameters are actually saved in ema field after train.

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Ema's parameter are updated with the formula:
           `ema_param = (1-momentum) * ema_param + momentum * cur_param`.
            Defaults to 0.0002.
        skip_buffers (bool): Whether to skip the model buffers, such as
            batchnorm running stats (running_mean, running_var), it does not
            perform the ema operation. Default to False.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        resume_from (str, optional): The checkpoint path. Defaults to None.
        momentum_fun (func, optional): The function to change momentum
            during early iteration (also warmup) to help early training.
            It uses `momentum` as a constant. Defaults to None.
    """

    def __init__(self,
                 momentum=0.0002,
                 interval=1,
                 skip_buffers=False,
                 resume_from=None,
                 momentum_fun=None,
                 load_type='straight',
                 ):
        assert 0 < momentum < 1
        self.momentum = momentum
        self.skip_buffers = skip_buffers
        self.interval = interval
        self.checkpoint = resume_from
        self.momentum_fun = momentum_fun
        self.load_type = load_type

    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as ``named_buffer`` to model.
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        self.param_ema_buffer = {}
        if self.skip_buffers:
            self.model_parameters = dict(model.named_parameters())
        else:
            self.model_parameters = model.state_dict()
        for name, value in self.model_parameters.items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers())
        if self.checkpoint is not None:
            runner.resume(self.checkpoint)

    def get_momentum(self, runner):
        return self.momentum_fun(runner.iter) if self.momentum_fun else \
                        self.momentum

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        if (runner.iter + 1) % self.interval != 0:
            return
        momentum = self.get_momentum(runner)
        for name, parameter in self.model_parameters.items():
            # exclude num_tracking
            if parameter.dtype.is_floating_point:
                buffer_name = self.param_ema_buffer[name]
                buffer_parameter = self.model_buffers[buffer_name]
                buffer_parameter.mul_(1 - momentum).add_(
                    parameter.data, alpha=momentum)
        if self.load_type == 'straight':
            from .load import load_state_dict
            load_state_dict(runner.eval_model, self.model_parameters)
        elif self.load_type == 'ema':
            # from IPython import embed
            # embed()
            # exit()
            for name, param_q in runner.eval_model.named_parameters():
                # print(name)
                buffer_name = self.param_ema_buffer[name[7:]]  # remove module.
                buffer_parameter = self.model_buffers[buffer_name]
                param_q.data = buffer_parameter
                # param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    def after_train_epoch(self, runner):
        """We load parameter values from ema backup to model before the
        EvalHook."""
        self._swap_ema_parameters()

    def before_train_epoch(self, runner):
        """We recover model's parameter from ema backup after last epoch's
        EvalHook."""
        self._swap_ema_parameters()

    def _swap_ema_parameters(self):
        """Swap the parameter of model with parameter in ema_buffer."""
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)


@HOOKS.register_module()
class CustomExpMomentumEMAHook(BaseEMAHook):
    """EMAHook using exponential momentum strategy.

    Args:
        total_iter (int): The total number of iterations of EMA momentum.
           Defaults to 2000.
    """

    def __init__(self, total_iter=2000, **kwargs):
        super(CustomExpMomentumEMAHook, self).__init__(**kwargs)
        self.momentum_fun = lambda x: (1 - self.momentum) * math.exp(-(
            1 + x) / total_iter) + self.momentum


@HOOKS.register_module()
class CustomLinearMomentumEMAHook(BaseEMAHook):
    """EMAHook using linear momentum strategy.

    Args:
        warm_up (int): During first warm_up steps, we may use smaller decay
            to update ema parameters more slowly. Defaults to 100.
    """

    def __init__(self, warm_up=100, **kwargs):
        super(CustomLinearMomentumEMAHook, self).__init__(**kwargs)
        self.momentum_fun = lambda x: min(self.momentum**self.interval,
                                          (1 + x) / (warm_up + x))
