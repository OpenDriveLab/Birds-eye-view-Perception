# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

__all__ = ['is_parallel']


def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)
