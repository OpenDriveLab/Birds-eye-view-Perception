_base_ = ['../../datasets/custom_waymo-3d.py', '../../../../mmdetection3d/configs/_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly

voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names = ['Car', 'Pedestrian', 'Cyclist']
point_cloud_range = [-35, -75, -3, 75, 75, 4]
input_modality = dict(use_lidar=False, use_camera=True)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 300
bev_w_ = 220
num_queue = 4

model = dict(
    type='BEV_Former',
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2_silent', deform_groups=1,
                 fallback_on_stride=False),  # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(type='FPN',
                  in_channels=[512, 1024, 2048],
                  out_channels=_dim_,
                  start_level=0,
                  add_extra_convs='on_output',
                  num_outs=4,
                  relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='BEV_FormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=3,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=8,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='BEVTransformer',
            rotate_prev_bev='waymo_rotate',
            use_can_bus=True,
            can_bus_norm=False,
            use_shift='waymo_shift',
            embed_dims=_dim_,
            num_cams=5,
            Z=7,
            encoder=dict(
                type='BEVTransformerEncoderV2',
                num_layers=3,
                pc_range=point_cloud_range,
                num_pre_bev_layer=3,
                use_key_padding_mask=True,
                return_intermediate=False,
                # dataset_type='waymo',
                transformerlayers=dict(type='BEVEncoderLayerV2',
                                       attn_cfgs=[
                                           dict(type='CustomMultiScaleDeformableAttentionV4',
                                                embed_dims=_dim_,
                                                num_levels=1),
                                           dict(
                                               type='BEVCrossDeformableAtten',
                                               pc_range=point_cloud_range,
                                               num_cams=5,
                                               deformable_attention=dict(type='MultiScaleDeformableAttention3D',
                                                                         embed_dims=_dim_,
                                                                         num_points=8,
                                                                         num_levels=_num_levels_),
                                               embed_dims=_dim_,
                                           )
                                       ],
                                       feedforward_channels=_ffn_dim_,
                                       ffn_dropout=0.1,
                                       operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(type='BEVTransformerDecoder',
                         num_layers=6,
                         return_intermediate=True,
                         transformerlayers=dict(type='DetrTransformerDecoderLayer',
                                                attn_cfgs=[
                                                    dict(type='MultiheadAttention',
                                                         embed_dims=_dim_,
                                                         num_heads=8,
                                                         dropout=0.1),
                                                    dict(type='CustomMultiScaleDeformableAttention',
                                                         embed_dims=_dim_,
                                                         num_levels=1),
                                                ],
                                                feedforward_channels=_ffn_dim_,
                                                ffn_dropout=0.1,
                                                operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn',
                                                                 'norm')))),
        bbox_coder=dict(type='NMSFreeCoder',
                        post_center_range=[-80, -80, -10.0, 80.0, 80.0, 10.0],
                        pc_range=point_cloud_range,
                        max_num=300,
                        voxel_size=voxel_size,
                        num_classes=3),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),  # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

dataset_type = 'WaymoDataset_videoV2'
data_root = 'data/waymo_mini/'
file_client_args = dict(backend='disk')
gt_bin_file = 'data/waymo_mini/gt.bin'

train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomHorizontalFlipMultiViewImage', dataset='waymo'),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.9, 1.0, 1.1]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size='same2max'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size=(1280, 1920)),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1920, 1280),
         pts_scale_ratio=1,
         flip=False,
         transforms=[
             dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
             dict(type='CustomCollect3D', keys=['img'])
         ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'waymo_mini_infos_train.pkl',
        calib_file=data_root + 'waymo_calibs.pkl',
        gt_bin_file=gt_bin_file,
        use_pkl_annos=True,
        split='training',
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False,
        queue_length=num_queue,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        pcd_limit_range=point_cloud_range,
        bev_size=(bev_h_, bev_w_),
        img_format='.jpg',
        # load one frame every five frames
        load_interval=1),
    val=dict(type=dataset_type,
             pipeline=test_pipeline,
             data_root=data_root,
             ann_file=data_root + 'waymo_infos_val.pkl',
             calib_file=data_root + 'waymo_calibs.pkl',
             gt_bin_file=gt_bin_file,
             use_pkl_annos=True,
             split='training',
             pcd_limit_range=point_cloud_range,
             bev_size=(bev_h_, bev_w_),
             classes=class_names,
             modality=input_modality,
             samples_per_gpu=1,
             img_format='.jpg',
             load_interval=1),
    test=dict(type=dataset_type,
              pipeline=test_pipeline,
              data_root=data_root,
              ann_file=data_root + 'waymo_infos_val.pkl',
              calib_file=data_root + 'waymo_calibs.pkl',
              gt_bin_file=gt_bin_file,
              use_pkl_annos=True,
              split='training',
              pcd_limit_range=point_cloud_range,
              bev_size=(bev_h_, bev_w_),
              classes=class_names,
              modality=input_modality,
              samples_per_gpu=1,
              img_format='.jpg',
              load_interval=1),
    shuffler_sampler=dict(type='OriginDistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

optimizer = dict(type='AdamW2',
                 lr=2e-4,
                 paramwise_cfg=dict(custom_keys={
                     'img_backbone': dict(lr_mult=0.1),
                 }),
                 weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=500, warmup_ratio=1.0 / 3, min_lr_ratio=1e-3)
total_epochs = 12
evaluation = dict(interval=12, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner_video', max_epochs=total_epochs)
load_from = 'ckpts/fcos3d.pth'

fp16 = dict(loss_scale=512.)
# fp16 =None
# find_unused_parameters = True
custom_hooks = [dict(type='TransferWeight', priority='LOWEST')]
# custom_hooks = [dict(type='GradChecker',priority='HIGHEST')]

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
checkpoint_config = dict(interval=1)
