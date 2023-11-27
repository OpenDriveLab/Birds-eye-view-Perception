from mmcv.utils import Registry, build_from_cfg
from torch import nn
import warnings
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

DISCRIMINATOR = Registry('discriminator')
DISTILLER = Registry('distiller')
DISTILL_LOSSES = Registry('distill_loss')


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_distill_loss(cfg):
    """Build distill loss."""
    return build(cfg, DISTILL_LOSSES)

def build_distiller(cfg,teacher_cfg=None, student_cfg=None, branch_distill_queries_cfg = None, train_cfg=None, test_cfg=None):
    """Build distiller."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return build(cfg, DISTILLER, dict(teacher_cfg=teacher_cfg, student_cfg=student_cfg, branch_distill_queries_cfg= branch_distill_queries_cfg))
