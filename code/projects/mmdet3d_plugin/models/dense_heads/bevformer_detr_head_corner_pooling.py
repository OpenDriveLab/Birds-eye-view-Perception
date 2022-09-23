# Copyright (c) 2022 OpenPerceptionX. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import CornerPool
from mmcv.runner import BaseModule
from mmcv.runner import auto_fp16
from mmdet.models import HEADS

from .bevformer_detr_head2 import BEV_FormerHeadV2
from .bevformer_centerpoint_head import BEV_FormerHead_centerpoint
from .bevformer_anchor_head import BEVFormer_FreeAnchor3DHead

__all__ = ['BEV_FormerHeadWithCornerPool']


@HEADS.register_module()
class BEV_FormerHeadWithCornerPool(BEV_FormerHeadV2):

    def __init__(self, *args, num_levels=4, **kwargs):
        self.num_levels = num_levels
        super(BEV_FormerHeadWithCornerPool, self).__init__(*args, **kwargs)

    def _init_layers(self):
        self.corner_pool = nn.ModuleList()
        for _ in range(self.num_levels):
            self.corner_pool.append(MyCornerPool(self.in_channels))
        super(BEV_FormerHeadWithCornerPool, self)._init_layers()

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, return_bev=False, gt_bboxes_3d=None, **kwargs):
        pooled_feats = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, nm, c, h, w = feat.shape
            pooled_feat = self.corner_pool[lvl](feat.view(bs * nm, c, h, w))
            pooled_feats.append(pooled_feat.view(bs, nm, c, h, w))
        return super(BEV_FormerHeadWithCornerPool, self).forward(pooled_feats, img_metas, prev_bev, return_bev,
                                                                 gt_bboxes_3d)


@HEADS.register_module()
class BEV_FormerHeadWithCornerPoolV2(BEV_FormerHead_centerpoint):

    def __init__(self, *args, num_levels=4, **kwargs):
        self.num_levels = num_levels
        super(BEV_FormerHeadWithCornerPoolV2, self).__init__(*args, **kwargs)

    def _init_layers(self):
        self.corner_pool = nn.ModuleList()
        for _ in range(self.num_levels):
            self.corner_pool.append(MyCornerPool(self.in_channels))
        super(BEV_FormerHeadWithCornerPoolV2, self)._init_layers()

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, return_bev=False, gt_bboxes_3d=None, **kwargs):
        pooled_feats = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, nm, c, h, w = feat.shape
            pooled_feat = self.corner_pool[lvl](feat.view(bs * nm, c, h, w))
            pooled_feats.append(pooled_feat.view(bs, nm, c, h, w))
        return super(BEV_FormerHeadWithCornerPoolV2, self).forward(pooled_feats, img_metas, prev_bev, return_bev,
                                                                   gt_bboxes_3d)


@HEADS.register_module()
class BEV_FormerHeadWithCornerPoolV4(BEVFormer_FreeAnchor3DHead):

    def __init__(self, *args, num_levels=4, **kwargs):
        self.num_levels = num_levels
        super(BEV_FormerHeadWithCornerPoolV4, self).__init__(*args, **kwargs)

    def _init_layers(self):
        self.corner_pool = nn.ModuleList()
        for _ in range(self.num_levels):
            self.corner_pool.append(MyCornerPool(self.in_channels))
        super(BEV_FormerHeadWithCornerPoolV4, self)._init_layers()

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, return_bev=False, gt_bboxes_3d=None, **kwargs):
        pooled_feats = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, nm, c, h, w = feat.shape
            pooled_feat = self.corner_pool[lvl](feat.view(bs * nm, c, h, w))
            pooled_feats.append(pooled_feat.view(bs, nm, c, h, w))
        return super(BEV_FormerHeadWithCornerPoolV4, self).forward(pooled_feats, img_metas, prev_bev, return_bev,
                                                                   gt_bboxes_3d)


class MyCornerPool(BaseModule):

    def __init__(self, in_channels, norm_cfg=dict(type='BN', requires_grad=True), init_cfg=None):
        super(MyCornerPool, self).__init__(init_cfg)

        self.top_pool = CornerPool('top')
        self.left_pool = CornerPool('left')
        self.bottom_pool = CornerPool('bottom')
        self.right_pool = CornerPool('right')
        self.aftpool_conv = ConvModule(in_channels * 2, in_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        top_feat = self.top_pool(x)
        left_feat = self.left_pool(x)
        bottom_feat = self.bottom_pool(x)
        right_feat = self.right_pool(x)
        aftpool_conv = self.aftpool_conv(torch.cat([top_feat + left_feat, bottom_feat + right_feat], dim=1))
        return aftpool_conv + x
