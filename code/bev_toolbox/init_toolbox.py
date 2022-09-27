import importlib


def init_toolbox_mmdet3d():
    '''
    Init wrappers for mmdet3d.
    '''
    importlib.import_module('toolbox.wrapper_mmdet3d')
