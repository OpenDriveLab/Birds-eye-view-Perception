# compared to bevformer_base, bevformer_tiny has
# smaller backbone: R101-DCN
# smaller BEV: 200*200 -> 100*100
# less encoder layers: 6 -> 3
# smaller input size: 1600*900 -> 800*450
# multi-scale feautres -> single scale features (C5)

_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    #mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False
)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True
    )

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 100
bev_w_ = 100

queue_length = 4 # each sequence contains `queue_length` frames.

# knowledge distillation settings
teacher_cfg  = 'projects/configs/bevformer/bevformer_base.py'
student_cfg  = 'projects/configs/bevformer/bevformer_small.py'
weight = 0.25
tau = 2.0

distiller = dict(
    type='HeatmapDetectionDistiller',
    teacher_pretrained =  'ckpts/bevformer_r101_dcn_24ep.pth',
    distill_on_FPN = True,
    distill_on_BEV = True,
    fpn_feats_norm = None,
    bev_feats_norm = None,
    duplicate_student_head = True,
    distiller_fpn = [ dict(student_module = 'img_neck.fpn_convs.3.conv',
                         teacher_module = 'img_neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='ChannelWiseDivergence',
                                       name='loss_cw_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       tau = tau,
                                       weight =weight,
                                       )
                                ]
                        ),
                    dict(student_module = 'img_neck.fpn_convs.2.conv',
                         teacher_module = 'img_neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='ChannelWiseDivergence',
                                       tau = tau,
                                       name='loss_cw_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       weight =weight,
                                       )
                                ]
                        ),
                    dict(student_module = 'img_neck.fpn_convs.1.conv',
                         teacher_module = 'img_neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='ChannelWiseDivergence',
                                       tau = tau,
                                       name='loss_cw_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       weight =weight,
                                       )
                                ]
                        ),
                    dict(student_module = 'img_neck.fpn_convs.0.conv',
                         teacher_module = 'img_neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='ChannelWiseDivergence',
                                       tau = tau,
                                       name='loss_cw_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       weight =weight,
                                       )
                                ]
                        ),
                   ],
    distiller_bev =   [ dict(student_module = 'bev',
                    teacher_module = 'bev',
                    output_hook = True,
                    methods=[dict(type= 'L2Loss',
                                name='loss_l2_bev',
                                student_channels = 256,
                                teacher_channels = 256,
                                weight = 0.1,
                                )
                            ])]
    )




dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size=(464, 800)), #size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    #dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size=(464, 800)), #size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu= 1,
    workers_per_gpu= 4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1,),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality,),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='CustomEpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
load_student = True
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)