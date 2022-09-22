# ==============================================================================
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
# ==============================================================================

import time
import copy
import numpy as np
import torch
from torch.nn import functional as F
import torch.utils.checkpoint as cp
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
import mmdet3d
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.models.utils.dummy_metas import dummy_metas, lss_dummy_metas


@DETECTORS.register_module()
class BEV_Former(MVXTwoStageDetector):
    """BEV_Former."""

    def __init__(
        self,
        use_lidar_teacher=False,
        use_grid_mask=False,
        use_checkpoint=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
    ):

        super(BEV_Former, self).__init__(pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder, pts_fusion_layer,
                                         img_backbone, pts_backbone, img_neck, pts_neck, pts_bbox_head, img_roi_head,
                                         img_rpn_head, train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.use_checkpoint = use_checkpoint
        self.fp16_enabled = False
        self.use_lidar_teacher = use_lidar_teacher
        self.video_test_mode = video_test_mode
        self.prev_bev = None
        self.prev_idx = None
        self.prev_pos = 0
        self.prev_angle = 9
        self.count = 0

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.
        Args:
            points (list[torch.Tensor]): Points of each sample.
        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @torch.no_grad()
    @force_fp32()
    def extract_pts_feat(self, pts, img_feats=None, img_metas=None):
        """Extract features of points."""

        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        # self.pts_backbone.fp16_enabled = False

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors, img_feats, img_metas)
        batch_size = coors[-1, 0] + 1

        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    # @auto_fp16(apply_to=('img'))
    # @run_time('extract_img_feat')
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            if self.training and self.use_checkpoint:
                img.requires_grad_(True)
                img_feats = cp.checkpoint(self.img_backbone, img)
            else:
                img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas)

        return img_feats

    # @run_time('forward_pts_train')
    def forward_pts_train(
        self,
        pts_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
        prev_bev=None,
        bev_teacher=None,
    ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        # from IPython import embed
        # embed()
        # exit()
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev, bev_teacher=bev_teacher, gt_bboxes_3d=gt_bboxes_3d)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas['lss_metas'] = lss_dummy_metas
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

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
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        if self.use_lidar_teacher and points is not None:
            with torch.no_grad():
                self.eval()
                lidar_bev = self.extract_pts_feat(points)
                self.train()
                # from IPython import embed
                # embed()
                # exit()
                # from projects.mmdet3d_plugin.models.utils.visual import save_tensor
                # save_tensor(lidar_bev[0][0, :100], f'lidar_bev_{self.count}.png')
                # self.count += 1
        else:
            lidar_bev = None

            # if prev_bev is not None:
            #     bev = prev_bev.reshape(300, 220, 256)
            #     bev = bev.permute(2, 0, 1)
            #     save_tensor(bev[:50], 'img_bev.png')

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
        losses.update(losses_pts)
        return losses

    # @run_time('forward_pts_test')
    def forward_test(self, img_metas, img=None, points=None, **kwargs):

        # lidar_bev = self.extract_pts_feat(points[0])
        # from projects.mmdet3d_plugin.models.utils.visual import save_tensor
        # from IPython import embed
        # embed()
        # exit()
        # save_tensor(lidar_bev[0][0, :100], f'lidar_bev_{self.count}.png')
        # self.count += 1

        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_idx:
            self.prev_bev = None  # the first sample of each scene or a scene is truncated

        self.prev_idx = img_metas[0][0]['scene_token']  # update idx

        if not self.video_test_mode:
            self.prev_bev = None  # signle sample test

        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_bev is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_pos
            img_metas[0][0]['can_bus'][-1] -= self.prev_angle
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0
        self.prev_pos = tmp_pos
        self.prev_angle = tmp_angle
        new_prev_bev, bbox_results = self.simple_test(img_metas[0], img[0], prev_bev=self.prev_bev, **kwargs)
        self.prev_bev = new_prev_bev
        results = {'bbox_results': bbox_results, 'mask_results': None}
        return results
        # if num_augs == 1:
        #     img = [img] if img is None else img
        #     return self.simple_test(None, img_metas[0], img[0], **kwargs)
        # else:
        #     return self.aug_test(None, img_metas, img, **kwargs)

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function of point cloud branch."""
        new_prev_bev, outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, return_bev=True)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
        return new_prev_bev, bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def val_step(self, data, optimizer):
        """The iteration step during validation.
        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """

        if data.get('return_bev', False):
            img = data['img']
            img_metas = data['img_metas']
            points = data.get('points', None)
            # print(img_metas)
            if self.use_lidar_teacher and points is not None:
                lidar_bev = self.extract_pts_feat(points)
            else:
                lidar_bev = None

            # print('kwargs[img]',kwargs['img'].data[0].shape,kwargs['img'].data[0].device)
            img_feats = self.extract_feat(img=img, img_metas=img_metas)
            prev_bev = data.get('prev_bev', None)
            outs = self.pts_bbox_head(img_feats, img_metas, bev_teacher=lidar_bev, prev_bev=prev_bev)
            # from IPython import embed
            # embed()
            # exit()
            return outs
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs