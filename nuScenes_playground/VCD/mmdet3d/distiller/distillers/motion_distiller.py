import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmdet3d.core import bbox3d2result
from mmcv.runner import  load_checkpoint
from ..builder import DISTILLER, build_distill_loss
import copy
import numpy as np
import cv2
import math
import mmcv
from mmcv import Config
from pyquaternion import Quaternion
#from nuscenes.utils.data_classes import Box
from nuscenes.utils.data_classes import Box as NuScenesBox
import matplotlib.pyplot as plt
from mmdet3d.core.bbox import LiDARInstance3DBoxes
import os
os.sys.path.append('.')


@DISTILLER.register_module()
class MotionDistiller(BaseDetector):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_head_cfg = None,
                 distill_on_PV = False,
                 distill_on_BEV = False,
                 distill_on_depth = False,
                 distill_PV = None,
                 distill_BEV = None,
                 distill_depth = None,
                 channel_PV = 64,
                 channel_BEV = 256,
                 channel_depth = 64,
                 motion_length = None,
                 downsample_align_mode = False,
                 feats_align_norm = None,
                 teacher_pretrained=None,
                 student_init=None,
                 duplicate_student_head = False,):

        super(MotionDistiller, self).__init__()
        self.teacher_cfg = teacher_cfg
        self.student_cfg = student_cfg
        self.teacher = build_detector(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.init_weights_teacher(teacher_pretrained)
        for (name, param) in self.teacher.named_parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.student = build_detector(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))
        if student_init is not None:
            self.init_weights_student(student_init)
        if duplicate_student_head:
            self.init_weights_student_head(teacher_pretrained)
        
        self.class_names = teacher_cfg.get('class_names')

        # loss items
        self.distill_on_PV = distill_on_PV
        if self.distill_on_PV:
            self.distill_losses_pv = nn.ModuleDict()
            self.distill_pv = distill_PV
            if distill_PV is not None:
                for item_loc in distill_PV:
                    for item_loss in item_loc.methods:
                        loss_name = item_loss.name
                        self.distill_losses_pv[loss_name] = build_distill_loss(item_loss)

        self.distill_on_BEV = distill_on_BEV
        if self.distill_on_BEV:
            self.distill_losses_bev = nn.ModuleDict()
            self.distill_bev = distill_BEV
            if distill_BEV is not None:
                for item_loc in distill_BEV:
                    for item_loss in item_loc.methods:
                        loss_name = item_loss.name
                        self.distill_losses_bev[loss_name] = build_distill_loss(item_loss)

        self.distill_on_depth = distill_on_depth
        if self.distill_on_depth:
            self.distill_losses_depth = nn.ModuleDict()
            self.distill_depth = distill_depth
            if distill_depth is not None:
                for item_loc in distill_depth:
                    for item_loss in item_loc.methods:
                        loss_name = item_loss.name
                        self.distill_losses_depth[loss_name] = build_distill_loss(item_loss)
        
        # # self.downsample_align_mode = downsample_align_mode
        # # if self.distill_on_PV and not downsample_align_mode:
        # if self.distill_on_PV:
        #     self.restore_layers_pv = self._make_deconv_layers(channel_PV, channel_PV, num_deconv_layers = 1)
        # #if self.distill_on_BEV and not downsample_align_mode:
        # if self.distill_on_BEV:
        #     self.restore_layers_bev = self._make_deconv_layers(channel_BEV, channel_BEV, num_deconv_layers = 1)
        # if self.distill_on_depth:
        #     self.restore_layers_depth = self._make_deconv_layers(channel_depth, channel_depth, num_deconv_layers = 1)
        self.flag_test_teacher = False
        self.feats_align_norm = feats_align_norm
        if self.feats_align_norm == 'ln':
            self.norm_align = nn.LayerNorm(channel_BEV)
        elif self.feats_align_norm == 'bn':
            self.norm_align = nn.BatchNorm2d(channel_BEV, affine=False, track_running_stats=False) 

        if motion_length is not None:
            self.motion_length = motion_length
        else:
            self.motion_length = None



    def base_parameters(self):
        return nn.ModuleList([self.student, self.distill_losses])

    def discriminator_parameters(self):
        return self.discriminator

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')
    
    def init_weights_student(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.student, path, map_location='cpu')

    def init_weights_student_head(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.student.pts_bbox_head, path, map_location='cpu', \
            revise_keys=[(r'^module\.', ''), (r'^pts_bbox_head\.', '')])

    def _make_deconv_layers(self, inplanes, outplanes, num_deconv_layers=1, add_extra_layer = True):
        def _get_deconv_cfg(deconv_kernel):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0
            return deconv_kernel, padding, output_padding
        
        layers = []
        for idx in range(num_deconv_layers):
            kernel, padding, output_padding = _get_deconv_cfg(4)
            layers.append(
                nn.ConvTranspose2d(
                    in_channels = inplanes,
                    out_channels = outplanes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias = False))
            # layers.append(nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM))
            if(add_extra_layer is False and idx == num_deconv_layers-1):
                break
            layers.append(nn.ReLU(inplace=True))
        
        if add_extra_layer:
            layers.append(
                nn.Conv2d(
                    in_channels=outplanes,
                    out_channels=outplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1))

        return nn.Sequential(*layers)

    def forward(self, return_loss=True, return_result=False, **kwargs):
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
            losses =  self.forward_train(**kwargs)
            return losses
        else:
            if not self.flag_test_teacher:
                return self.forward_test(**kwargs)
            else:
                return self.forward_test_teacher(**kwargs)

    def forward_train(self, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """
        # inputs = kwargs['img_inputs']
        # B, N, _, H, W = inputs[0].shape
        # N_cam = 6
        # self.num_frame = N // N_cam
        # imgs = inputs[0].view(B, N_cam, self.num_frame, 3, H, W)
        # rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        # extra = [
        #     rots.view(B, self.num_frame, N, 3, 3),
        #     trans.view(B, self.num_frame, N, 3),
        #     intrins.view(B, self.num_frame, N, 3, 3),
        #     post_rots.view(B, self.num_frame, N, 3, 3),
        #     post_trans.view(B, self.num_frame, N, 3)
        # ]
        # def split_img_inputs(imgs, extra, start, end):
        #     for item in extra:
        #         item = item[:,start:end]
        #     return imgs[:,:,start:end], extra[0], extra[1], extra[2], extra[3], extra[4], bda
        N = kwargs['img_inputs'][0].shape[1]
        N_cam = 6
        self.num_frame = N // N_cam
            
        with torch.no_grad():
            self.teacher.eval()
            teacher_loss, bev_feats_tch, img_feats_tch, pred_depth_tch, depth_fusion_tch = self.teacher.forward_train(**kwargs, flag_distill=True)
        
        student_loss, bev_feats_stu, img_feats_stu, pred_depth_stu, depth_fusion_stu = self.student.forward_train(**kwargs, flag_distill=True)

        heatmaps = []
        heatmap_curr, anno_boxes, inds, masks = self.teacher.pts_bbox_head.get_targets(kwargs['gt_bboxes_3d'], kwargs['gt_labels_3d'])
        heatmaps.append(torch.cat(heatmap_curr, dim=1))
        
        indexes = kwargs['adj_index']
        indexes = torch.cumsum(indexes, dim=1)
        gt_bboxes_3d_adj = kwargs['gt_bboxes_3d_adj']
        gt_labels_3d_adj = kwargs['gt_labels_3d_adj']
        for idx in range(1, indexes.shape[1]):
            gt_boxes = self.single_adj_objects(gt_bboxes_3d_adj, indexes[:, idx-1], indexes[:, idx])
            gt_labels = self.single_adj_objects(gt_labels_3d_adj, indexes[:, idx-1], indexes[:, idx])

            heatmap_adj, anno_boxes, inds, masks = self.teacher.pts_bbox_head.get_targets(gt_boxes, gt_labels)
            heatmaps.append(torch.cat(heatmap_adj, dim=1))
        
        _, rots, trans, _, _, _, bda = self.prepare_inputs(kwargs['img_inputs'])
        for adj_id in range(1, self.num_frame):
            heatmaps[adj_id] = \
                self.teacher.shift_feature(heatmaps[adj_id],
                                    [trans[0], trans[adj_id]],
                                    [rots[0], rots[adj_id]],
                                    bda)

        # if self.motion_length is not None:
        #     heatmaps = heatmaps[-self.motion_length:]
        mask_heatmap = torch.cat(heatmaps, dim=1).max(dim=1).values.unsqueeze(1)
        for k, v in teacher_loss.items():
            student_loss[k+'_T'] = v
        # distill on PV features
        if self.distill_on_PV:
            img_feats_stu = self.restore_layers_pv(img_feats_stu)
            if self.distill_pv is not None:
                for item_loc in self.distill_pv:
                    for item_loss in item_loc.methods:
                        loss_name = item_loss.name
                        student_loss[loss_name] = self.distill_losses_pv[loss_name](img_feats_stu, img_feats_tch)
        # distill on BEV features
        if self.distill_on_BEV:
            # bev_feats_stu = self.restore_layers_bev(bev_feats_stu)
            if self.feats_align_norm is not None:
                bev_feats_tch = self.norm_align(bev_feats_tch)
            bev_feats_stu = bev_feats_stu * mask_heatmap
            bev_feats_tch = bev_feats_tch * mask_heatmap
            h_T, w_T = bev_feats_tch.shape[-2:]
            if self.distill_bev is not None:
                for item_loc in self.distill_bev:
                    for item_loss in item_loc.methods:
                        loss_name = item_loss.name
                        student_loss[loss_name] = (1/mask_heatmap.sum()) * self.distill_losses_bev[loss_name](bev_feats_stu, bev_feats_tch)

        if self.distill_on_depth:
            pred_depth_stu = self.restore_layers_depth(pred_depth_stu)
            if self.distill_depth is not None:
                for item_loc in self.distill_depth:
                    for item_loss in item_loc.methods:
                        loss_name = item_loss.name
                        student_loss[loss_name] = self.distill_losses_depth[loss_name](pred_depth_stu, pred_depth_tch)
        return student_loss
    
    def single_adj_objects(self, inputs, starts, ends):
        B = len(inputs)
        output = []
        for i in range(B):
            gt_adj = inputs[i]
            start = starts[i]
            end = ends[i]
            if isinstance(gt_adj, LiDARInstance3DBoxes):
                gt_adj = gt_adj.tensor
                adj_objects = gt_adj[start:end]
                adj_objects = LiDARInstance3DBoxes(adj_objects, box_dim=adj_objects.shape[-1],
                                                origin=(0.5, 0.5, 0.5))
            else:
                adj_objects = gt_adj[start:end]
            output.append(adj_objects)
        return output

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        B, N, _, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        extra = [
            rots.view(B, self.num_frame, N, 3, 3),
            trans.view(B, self.num_frame, N, 3),
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        return imgs, rots, trans, intrins, post_rots, post_trans, bda

    def forward_test(self, **kwargs):
        return self.student.forward_test(**kwargs)
    
    def forward_test_teacher(self, **kwargs):
        return self.teacher.forward_test(**kwargs)

    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)

    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)

