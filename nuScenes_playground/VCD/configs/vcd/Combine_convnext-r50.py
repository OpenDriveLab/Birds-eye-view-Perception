# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = ['../_base_/datasets/nus-3d.py',
          '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

num_gpus = 8
samples_per_gpu = 8
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu) * 4.554)
num_epochs = 20
checkpoint_epoch_interval = 1
use_custom_eval_hook_for_distill=True
# use_custom_eval_hook = True
train_sequences_split_num = 2
test_sequences_split_num = 1
filter_empty_gt = False

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80

multi_adj_frame_id_cfg = (1, 4+1, 1)
key_frames = 0

# knowledge distillation settings
teacher_cfg  = 'configs/vcd/vcd-convnext-base.py'
student_cfg  = 'configs/vcd/vcd-r50.py'
weight = 0.0000001
tau = 1.0
_channel_PV = 64
_channel_BEV = 256
_channel_depth = 118
distiller = dict(
    type='CombineDistiller',
    teacher_pretrained = 'ckpts/teacher/iter_8341_ema.pth',
    student_init = None,
    channel_PV = _channel_PV,
    channel_BEV = _channel_BEV,
    channel_depth = _channel_depth,
    distill_on_PV = False,
    distill_on_BEV = True,
    distill_on_depth = False,
    distill_on_occupancy = True,
    feats_align_norm = 'bn',
    occ_resolution = {'x': [-51.2, 51.2, 0.4], 'y': [-51.2, 51.2, 0.4], 'z': [-5, 3, 1]},
    distill_PV =   [ dict(student_module = 'pv',
                    teacher_module = 'pv',
                    output_hook = True,
                    methods=[dict(type= 'L2Loss',
                                name='loss_l2_pv',
                                student_channels = _channel_PV,
                                teacher_channels = _channel_PV,
                                weight = weight*0.01,
                                )
                            ])],
    distill_BEV =   [ dict(student_module = 'bev',
                    teacher_module = 'bev',
                    output_hook = True,
                    methods=[dict(type= 'L2Loss',
                                name='loss_l2_bev',
                                student_channels = _channel_BEV,
                                teacher_channels = _channel_BEV,
                                weight = 0.05 * 8,
                                )
                            ])],
    distill_depth =   [ dict(student_module = 'depth',
                teacher_module = 'depth',
                output_hook = True,
                methods=[dict(type= 'KLLoss',
                            name='loss_kl_depth',
                            student_channels = _channel_depth,
                            teacher_channels = _channel_depth,
                            tau = 1.0,
                            weight = weight*40,
                            )
                        ])],
    distill_occupancy =   [ dict(student_module = 'occ',
                    teacher_module = 'occ',
                    output_hook = True,
                    methods=[dict(type= 'L1Loss',
                                name='loss_l1_occ',
                                student_channels = 8,
                                teacher_channels = 8,
                                weight = 0.05 * 8 * 5 * 2, # 0.001,
                                )
                            ])],
    )


# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
#file_client_args = dict(backend='disk')
file_client_args = dict(
    backend='disk',
    )

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        # sequential=True,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadMotionTrajectory'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
                                'gt_depth', 'gt_bboxes_3d_adj', 'gt_labels_3d_adj', 'adj_index'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=False, file_client_args=file_client_args),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_depth'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    key_frames=key_frames,
    use_sequence_group_flag=True,
)


test_data_config = dict(
    pipeline=test_pipeline,
    data_root = data_root,
    sequences_split_num=test_sequences_split_num,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=8,
    test_dataloader=dict(runner_type='EpochBasedRunner'),
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        modality=input_modality,
        img_info_prototype='bevdet',
        sequences_split_num=train_sequences_split_num,
        use_sequence_group_flag=True,
        filter_empty_gt=filter_empty_gt,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
# data['train']['dataset'].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=1.6e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[num_iters_per_epoch*num_epochs,])
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
checkpoint_config = dict(
    interval=checkpoint_epoch_interval * num_iters_per_epoch)
evaluation = dict(
    interval= num_iters_per_epoch, pipeline=test_pipeline)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=checkpoint_epoch_interval*num_iters_per_epoch,
    ),
    dict(
        type='SequentialControlDistillHook',
        temporal_start_iter=num_iters_per_epoch * 2,
    ),
]
log_config = dict(
    interval=50,
    hooks=[
        # dict(type='WechatLoggerHook'),
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# fp16 = dict(loss_scale='dynamic')

# dist_params = dict(backend='nccl', port=1234)
