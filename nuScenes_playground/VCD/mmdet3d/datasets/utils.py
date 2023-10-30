# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

# yapf: disable
from mmdet3d.datasets.pipelines import (Collect3D, DefaultFormatBundle3D,
                                        LoadAnnotations3D,
                                        LoadImageFromFileMono3D,
                                        LoadMultiViewImageFromFiles,
                                        LoadPointsFromFile,
                                        LoadPointsFromMultiSweeps,
                                        MultiScaleFlipAug3D,
                                        PointSegClassMapping)
from mmdet.datasets.pipelines import LoadImageFromFile, MultiScaleFlipAug
# yapf: enable
from .builder import PIPELINES


def is_loading_function(transform):
    """Judge whether a transform function is a loading function.

    Note: `MultiScaleFlipAug3D` is a wrapper for multiple pipeline functions,
    so we need to search if its inner transforms contain any loading function.

    Args:
        transform (dict | :obj:`Pipeline`): A transform config or a function.

    Returns:
        bool: Whether it is a loading function. None means can't judge.
            When transform is `MultiScaleFlipAug3D`, we return None.
    """
    # TODO: use more elegant way to distinguish loading modules
    loading_functions = (LoadImageFromFile, LoadPointsFromFile,
                         LoadAnnotations3D, LoadMultiViewImageFromFiles,
                         LoadPointsFromMultiSweeps, DefaultFormatBundle3D,
                         Collect3D, LoadImageFromFileMono3D,
                         PointSegClassMapping)
    if isinstance(transform, dict):
        obj_cls = PIPELINES.get(transform['type'])
        if obj_cls is None:
            return False
        if obj_cls in loading_functions:
            return True
        if obj_cls in (MultiScaleFlipAug3D, MultiScaleFlipAug):
            return None
    elif callable(transform):
        if isinstance(transform, loading_functions):
            return True
        if isinstance(transform, (MultiScaleFlipAug3D, MultiScaleFlipAug)):
            return None
    return False


def get_loading_pipeline(pipeline):
    """Only keep loading image, points and annotations related configuration.

    Args:
        pipeline (list[dict] | list[:obj:`Pipeline`]):
            Data pipeline configs or list of pipeline functions.

    Returns:
        list[dict] | list[:obj:`Pipeline`]): The new pipeline list with only
            keep loading image, points and annotations related configuration.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadPointsFromFile',
        ...         coord_type='LIDAR', load_dim=4, use_dim=4),
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations3D',
        ...         with_bbox=True, with_label_3d=True),
        ...    dict(type='Resize',
        ...         img_scale=[(640, 192), (2560, 768)], keep_ratio=True),
        ...    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        ...    dict(type='PointsRangeFilter',
        ...         point_cloud_range=point_cloud_range),
        ...    dict(type='ObjectRangeFilter',
        ...         point_cloud_range=point_cloud_range),
        ...    dict(type='PointShuffle'),
        ...    dict(type='Normalize', **img_norm_cfg),
        ...    dict(type='Pad', size_divisor=32),
        ...    dict(type='DefaultFormatBundle3D', class_names=class_names),
        ...    dict(type='Collect3D',
        ...         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadPointsFromFile',
        ...         coord_type='LIDAR', load_dim=4, use_dim=4),
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations3D',
        ...         with_bbox=True, with_label_3d=True),
        ...    dict(type='DefaultFormatBundle3D', class_names=class_names),
        ...    dict(type='Collect3D',
        ...         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ...    ]
        >>> assert expected_pipelines == \
        ...        get_loading_pipeline(pipelines)
    """
    loading_pipeline = []
    for transform in pipeline:
        is_loading = is_loading_function(transform)
        if is_loading is None:  # MultiScaleFlipAug3D
            # extract its inner pipeline
            if isinstance(transform, dict):
                inner_pipeline = transform.get('transforms', [])
            else:
                inner_pipeline = transform.transforms.transforms
            loading_pipeline.extend(get_loading_pipeline(inner_pipeline))
        elif is_loading:
            loading_pipeline.append(transform)
    assert len(loading_pipeline) > 0, \
        'The data pipeline in your config file must include ' \
        'loading step.'
    return loading_pipeline


def extract_result_dict(results, key):
    """Extract and return the data corresponding to key in result dict.

    ``results`` is a dict output from `pipeline(input_dict)`, which is the
        loaded data from ``Dataset`` class.
    The data terms inside may be wrapped in list, tuple and DataContainer, so
        this function essentially extracts data from these wrappers.

    Args:
        results (dict): Data loaded using pipeline.
        key (str): Key of the desired data.

    Returns:
        np.ndarray | torch.Tensor: Data term.
    """
    if key not in results.keys():
        return None
    # results[key] may be data or list[data] or tuple[data]
    # data may be wrapped inside DataContainer
    data = results[key]
    if isinstance(data, (list, tuple)):
        data = data[0]
    if isinstance(data, mmcv.parallel.DataContainer):
        data = data._data
    return data

  
import numpy as np
from pyquaternion import Quaternion

def nuscenes_get_rt_matrix(
    src_sample,
    dest_sample,
    src_mod,
    dest_mod):
    
    """
    CAM_FRONT_XYD indicates going from 2d image coords + depth
        Note that image coords need to multiplied with said depths first to bring it into 2d hom coords.
    CAM_FRONT indicates going from camera coordinates xyz
    
    Method is: whatever the input is, transform to global first.
    """
    possible_mods = ['CAM_FRONT_XYD', 
                     'CAM_FRONT_RIGHT_XYD', 
                     'CAM_FRONT_LEFT_XYD', 
                     'CAM_BACK_XYD', 
                     'CAM_BACK_LEFT_XYD', 
                     'CAM_BACK_RIGHT_XYD',
                     'CAM_FRONT', 
                     'CAM_FRONT_RIGHT', 
                     'CAM_FRONT_LEFT', 
                     'CAM_BACK', 
                     'CAM_BACK_LEFT', 
                     'CAM_BACK_RIGHT',
                     'lidar',
                     'ego',
                     'global']

    assert src_mod in possible_mods and dest_mod in possible_mods
    
    src_lidar_to_ego = np.eye(4, 4)
    src_lidar_to_ego[:3, :3] = Quaternion(src_sample['lidar2ego_rotation']).rotation_matrix
    src_lidar_to_ego[:3, 3] = np.array(src_sample['lidar2ego_translation'])
    
    src_ego_to_global = np.eye(4, 4)
    src_ego_to_global[:3, :3] = Quaternion(src_sample['ego2global_rotation']).rotation_matrix
    src_ego_to_global[:3, 3] = np.array(src_sample['ego2global_translation'])
    
    dest_lidar_to_ego = np.eye(4, 4)
    dest_lidar_to_ego[:3, :3] = Quaternion(dest_sample['lidar2ego_rotation']).rotation_matrix
    dest_lidar_to_ego[:3, 3] = np.array(dest_sample['lidar2ego_translation'])
    
    dest_ego_to_global = np.eye(4, 4)
    dest_ego_to_global[:3, :3] = Quaternion(dest_sample['ego2global_rotation']).rotation_matrix
    dest_ego_to_global[:3, 3] = np.array(dest_sample['ego2global_translation'])
    
    src_mod_to_global = None
    dest_global_to_mod = None
    
    if src_mod == "global":
        src_mod_to_global = np.eye(4, 4)
    elif src_mod == "ego":
        src_mod_to_global = src_ego_to_global
    elif src_mod == "lidar":
        src_mod_to_global = src_ego_to_global @ src_lidar_to_ego
    elif "CAM" in src_mod:
        src_sample_cam = src_sample['cams'][src_mod.replace("_XYD", "")]
        
        src_cam_to_lidar = np.eye(4, 4)
        src_cam_to_lidar[:3, :3] = src_sample_cam['sensor2lidar_rotation']
        src_cam_to_lidar[:3, 3] = src_sample_cam['sensor2lidar_translation']
        
        src_cam_intrinsics = np.eye(4, 4)
        src_cam_intrinsics[:3, :3] = src_sample_cam['cam_intrinsic']
        
        if "XYD" not in src_mod:
            src_mod_to_global = (src_ego_to_global @ src_lidar_to_ego @ 
                                 src_cam_to_lidar)
        else:
            src_mod_to_global = (src_ego_to_global @ src_lidar_to_ego @ 
                                 src_cam_to_lidar @ np.linalg.inv(src_cam_intrinsics))
            
            
    
    if dest_mod == "global":
        dest_global_to_mod = np.eye(4, 4)
    elif dest_mod == "ego":
        dest_global_to_mod = np.linalg.inv(dest_ego_to_global)
    elif dest_mod == "lidar":
        dest_global_to_mod = np.linalg.inv(dest_ego_to_global @ dest_lidar_to_ego)
    elif "CAM" in dest_mod:
        dest_sample_cam = dest_sample['cams'][dest_mod.replace("_XYD", "")]
        
        dest_cam_to_lidar = np.eye(4, 4)
        dest_cam_to_lidar[:3, :3] = dest_sample_cam['sensor2lidar_rotation']
        dest_cam_to_lidar[:3, 3] = dest_sample_cam['sensor2lidar_translation']
        
        dest_cam_intrinsics = np.eye(4, 4)
        dest_cam_intrinsics[:3, :3] = dest_sample_cam['cam_intrinsic']
        
        if "XYD" not in dest_mod:
            dest_global_to_mod = np.linalg.inv(dest_ego_to_global @ dest_lidar_to_ego @ 
                                               dest_cam_to_lidar)
        else:
            dest_global_to_mod = np.linalg.inv(dest_ego_to_global @ dest_lidar_to_ego @ 
                                               dest_cam_to_lidar @ np.linalg.inv(dest_cam_intrinsics))
    
    return dest_global_to_mod @ src_mod_to_global