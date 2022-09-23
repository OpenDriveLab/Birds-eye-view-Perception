# ==============================================================================
# Binaries and/or source for the following packages or projects are presented under one or more of the following open
# source licenses:
# bevformer_2d_aux.py       OpenPerceptionX        Apache License, Version 2.0
#
# Contact simachonghao@pjlab.org.cn if you have any issue
#
# See:
# https://github.com/zhiqi-li/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/detectors/bevformer.py
#
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
from torch import nn
import torch.utils.checkpoint as cp
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head
from .bevformer import BEV_Former


@DETECTORS.register_module()
class BEV_Former_2d_Aux(BEV_Former):
    """BEV_Former."""

    def __init__(
        self,
        bbox_head=None,
        learnable_loss_weight=False,
        learnable_loss_weight_p=0.5,
        *args,
        **kwargs,
    ):

        super(BEV_Former_2d_Aux, self).__init__(*args, **kwargs)

        self.bbox_head = build_head(bbox_head)

        self.learnable_loss_weight = learnable_loss_weight
        if self.learnable_loss_weight:
            self.learnable_loss_weight_p = learnable_loss_weight_p
            self.loss_weights = nn.Parameter(torch.ones(2), requires_grad=True)

    @auto_fp16(apply_to=('img', 'prev_bev', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      prev_bev=None):

        if self.use_lidar_teacher and points is not None:
            with torch.no_grad():
                self.eval()
                lidar_bev = self.extract_pts_feat(points)
                self.train()
        else:
            lidar_bev = None

        if img_metas[0]['prev_bev']:
            self.prev_bev = prev_bev
        else:
            self.prev_bev = None

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(img_feats,
                                            gt_bboxes_3d,
                                            gt_labels_3d,
                                            img_metas,
                                            gt_bboxes_ignore,
                                            self.prev_bev,
                                            bev_teacher=lidar_bev)
        for key in losses_pts:
            losses[key] = losses_pts[key] * \
                (0.5 / (self.loss_weights[0] ** 2) if self.learnable_loss_weight else 1)

        img_feats_reshaped = []
        for img_feat in img_feats:
            B, N, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B * N, C, H, W))
        losses_2d_aux = self.bbox_head.forward_train(img_feats_reshaped, img_metas, gt_bboxes[0], gt_labels[0], None)

        for key in losses_2d_aux:
            losses[key + '_2d'] = losses_2d_aux[key] * \
                (0.5 / (self.loss_weights[1] ** 2) if self.learnable_loss_weight else 1)

        if self.learnable_loss_weight:
            losses['learnable_loss'] = torch.log(1 + self.loss_weights**2).sum() * self.learnable_loss_weight_p

        return losses