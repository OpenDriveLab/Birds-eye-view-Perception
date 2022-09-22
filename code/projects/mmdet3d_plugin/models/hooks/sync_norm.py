# Copyright (c) OpenMMLab. All rights reserved.
# from ..utils.dist_utils import all_reduce_dict
import functools
import pickle
import warnings
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner import OptimizerHook, get_dist_info



def obj2tensor(pyobj, device='cuda'):
    """Serialize picklable python object to tensor."""
    storage = torch.ByteStorage.from_buffer(pickle.dumps(pyobj))
    return torch.ByteTensor(storage).to(device=device)


def tensor2obj(tensor):
    """Deserialize tensor to picklable python object."""
    return pickle.loads(tensor.cpu().numpy().tobytes())


def all_reduce_dict(py_dict, op='sum', group=None, to_float=True):
    """Apply all reduce function for python dict object.
    The code is modified from https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/utils/allreduce_norm.py.
    NOTE: make sure that py_dict in different ranks has the same keys and
    the values should be in the same shape. Currently only supports
    nccl backend.
    Args:
        py_dict (dict): Dict to be applied all reduce op.
        op (str): Operator, could be 'sum' or 'mean'. Default: 'sum'
        group (:obj:`torch.distributed.group`, optional): Distributed group,
            Default: None.
        to_float (bool): Whether to convert all values of dict to float.
            Default: True.
    Returns:
        OrderedDict: reduced python dict object.
    """
    warnings.warn(
        'group` is deprecated. Currently only supports NCCL backend.')
    _, world_size = get_dist_info()
    if world_size == 1:
        return py_dict

    # all reduce logic across different devices.
    py_key = list(py_dict.keys())
    if not isinstance(py_dict, OrderedDict):
        py_key_tensor = obj2tensor(py_key)
        dist.broadcast(py_key_tensor, src=0)
        py_key = tensor2obj(py_key_tensor)

    tensor_shapes = [py_dict[k].shape for k in py_key]
    tensor_numels = [py_dict[k].numel() for k in py_key]

    if to_float:
        warnings.warn('Note: the "to_float" is True, you need to '
                      'ensure that the behavior is reasonable.')
        flatten_tensor = torch.cat(
            [py_dict[k].flatten().float() for k in py_key])
    else:
        flatten_tensor = torch.cat([py_dict[k].flatten() for k in py_key])

    dist.all_reduce(flatten_tensor, op=dist.ReduceOp.SUM)
    if op == 'mean':
        flatten_tensor /= world_size

    split_tensors = [
        x.reshape(shape) for x, shape in zip(
            torch.split(flatten_tensor, tensor_numels), tensor_shapes)
    ]
    out_dict = {k: v for k, v in zip(py_key, split_tensors)}
    if isinstance(py_dict, OrderedDict):
        out_dict = OrderedDict(out_dict)
    return out_dict

def get_norm_states(module):
    async_norm_states = OrderedDict()
    for name, child in module.named_modules():
        if isinstance(child, nn.modules.batchnorm._NormBase):
            for k, v in child.state_dict().items():
                async_norm_states['.'.join([name, k])] = v
    return async_norm_states


@HOOKS.register_module()
class CustomSyncNormHook(Hook):
    """Synchronize Norm states after training epoch, currently used in YOLOX.
    Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to switch to synchronizing norm interval. Default: 15.
        interval (int): Synchronizing norm interval. Default: 1.
    """

    def __init__(self, num_last_epochs=15, interval=1):
        self.interval = interval
        self.num_last_epochs = num_last_epochs

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        if (epoch + 1) == runner.max_epochs - self.num_last_epochs:
            # Synchronize norm every epoch.
            self.interval = 1

    def after_train_epoch(self, runner):
        """Synchronizing norm."""
        epoch = runner.epoch
        module = runner.model
        if (epoch + 1) % self.interval == 0:
            _, world_size = get_dist_info()
            if world_size == 1:
                return
            norm_states = get_norm_states(module)
            if len(norm_states) == 0:
                return
            norm_states = all_reduce_dict(norm_states, op='mean')
            module.load_state_dict(norm_states, strict=False)