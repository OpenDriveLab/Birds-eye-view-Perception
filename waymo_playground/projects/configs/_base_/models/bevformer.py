_base_ = ['../datasets/custom_waymo-3d.py', '../default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

_dim_ = 256

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
)

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
runner = dict(type='EpochBasedRunner_video', max_epochs=total_epochs)
custom_hooks = [dict(type='TransferWeight', priority='LOWEST')]
