_base_ = ['../waymo_mini_r101_baseline.py']

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-35, -75, -3, 75, 75, 4]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
class_names = ['Car', 'Pedestrian', 'Cyclist']

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
