import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import  load_checkpoint
from ..builder import DISTILLER, build_distill_loss
import os
import cv2
import copy
import numpy as np
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion
#from tools.analysis_tools.visual import *


@DISTILLER.register_module()
class HeatmapDetectionDistiller(BaseDetector):
    """Base distiller for detectors.
    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_on_FPN = True,
                 distill_on_BEV = True,
                 distiller_fpn = None,
                 distiller_bev = None,
                 fpn_feats_norm = None,
                 bev_feats_norm = 'bn',
                 teacher_pretrained=None,
                 branch_distill_queries_cfg = None,
                 duplicate_student_head = False,
                 ):

        super(HeatmapDetectionDistiller, self).__init__()
        self.teacher_cfg = teacher_cfg
        self.student_cfg = student_cfg
        self.teacher = build_detector(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.bev_resolution_tch = (teacher_cfg.get('bev_h_'), teacher_cfg.get('bev_w_'))
        self.channel_tch = teacher_cfg.get('_dim_')
        self.init_weights_teacher(teacher_pretrained)
        for (name, param) in self.teacher.named_parameters():
            param.requires_grad = False
        self.teacher.eval()

        self.student = build_detector(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))

        if duplicate_student_head:
            print('duplicate_student_head')
            self.init_weights_student_head(teacher_pretrained)
        
        self.bev_resolution_stu = (student_cfg.get('bev_h_'), student_cfg.get('bev_w_'))
        self.channel_stu = student_cfg.get('_dim_')
        self.flag_distill_on_FPN = distill_on_FPN
        self.flag_distill_on_BEV = distill_on_BEV

        def regitster_hooks(student_module, teacher_module):
            def hook_teacher_forward(module, input, output):
                #self.register_buffer(teacher_module, output)
                setattr(self, teacher_module, output)
            def hook_student_forward(module, input, output):
                #self.register_buffer(student_module, output)
                setattr(self, student_module, output)
            return hook_teacher_forward, hook_student_forward
        
        modules_stu = dict(self.student.named_modules())
        modules_tch = dict(self.teacher.named_modules())

        # loss items
        self.distill_losses = nn.ModuleDict()

        self.distiller_fpn = distiller_fpn
        self.distiller_bev = distiller_bev

        if self.flag_distill_on_FPN:
            for item_loc in distiller_fpn:
                student_module = 'student_' + item_loc.student_module.replace('.','_')
                teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
                self.register_buffer(student_module, None)
                self.register_buffer(teacher_module, None)
                hook_teacher_forward, hook_student_forward = regitster_hooks(student_module, teacher_module)
                modules_tch[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
                modules_stu[item_loc.student_module].register_forward_hook(hook_student_forward)
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    self.distill_losses[loss_name] = build_distill_loss(item_loss)

        # BEV features alignment and distillation
        if self.flag_distill_on_BEV:
            self.bev_restore_layers = None
            if self.bev_resolution_tch != self.bev_resolution_stu:
                self.bev_restore_layers = self._make_deconv_layers(student_cfg._dim_, teacher_cfg._dim_, \
                    num_deconv_layers = int(self.bev_resolution_tch[0] / self.bev_resolution_stu[0] // 2))
            self.norm_bevFeats = None
            if bev_feats_norm == 'bn':
                self.norm_bevFeats = nn.BatchNorm2d(self.channel_tch, affine=False, track_running_stats=False)
            for item_loc in self.distiller_bev:
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    self.distill_losses[loss_name] = build_distill_loss(item_loss)
            



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

    def init_weights_student_head(self, path=None):
        """Load the pretrained model in teacher detector.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.student.pts_bbox_head, path, map_location='cpu', \
            revise_keys=[(r'^module\.', ''), (r'^pts_bbox_head\.', '')])

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

    def _make_deconv_layers(self, inplanes, outplanes, num_deconv_layers=1):
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
        BN_MOMENTUM = 0.1

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
            layers.append(nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))

        layers.append(
                nn.Conv2d(
                    in_channels=outplanes,
                    out_channels=outplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1))

        return nn.Sequential(*layers) 

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
        bs, seq_len, N_cam, c, h, w = kwargs['img'].shape
        kwargs_T = copy.deepcopy(kwargs)
        _scale = self.teacher_cfg.image_size[1] // self.student_cfg.image_size[1]
        if _scale > 1:
            kwargs_T['img'] = F.interpolate(kwargs['img'].view(bs * seq_len * N_cam, c, h, w), scale_factor=_scale, \
                    mode='bilinear', align_corners=False).view(bs, seq_len, N_cam, c, h*_scale, w*_scale)
            scale_factor = np.eye(4)
            scale_factor[0, 0] *= _scale
            scale_factor[1, 1] *= _scale
            for i_batch in range(len(kwargs_T['img_metas'])):
                for i_seq in kwargs_T['img_metas'][0].keys():
                    raw_img_metas = kwargs['img_metas'][i_batch][i_seq]
                    N_cam = len(raw_img_metas['filename'])
                    kwargs_T['img_metas'][i_batch][i_seq]['img_shape'] = [(img_shapes[0]*_scale, img_shapes[1]*_scale) for img_shapes in raw_img_metas['img_shape']]
                    kwargs_T['img_metas'][i_batch][i_seq]['pad_shape'] = [(img_shapes[0]*_scale, img_shapes[1]*_scale) for img_shapes in raw_img_metas['pad_shape']]
                    kwargs_T['img_metas'][i_batch][i_seq]['lidar2img'] = [scale_factor @ mats_lidar2img for mats_lidar2img in raw_img_metas['lidar2img']]
                

        # align student BEV feature with teacher BEV feature
        with torch.no_grad():
            self.teacher.eval()
            if self.flag_distill_on_FPN and not self.flag_distill_on_BEV:
                self.teacher.obtainImageViewFeats(**kwargs_T)
            elif self.flag_distill_on_BEV:
                bevFeats_tch = self.teacher.obtainBEVFeats(**kwargs_T)
                bevFeats_tch = bevFeats_tch.view(bevFeats_tch.shape[0], self.bev_resolution_tch[0], \
                    self.bev_resolution_tch[1], bevFeats_tch.shape[-1]).permute(0,3,1,2)
        student_output, student_loss = self.student.forward_train(**kwargs)
        bevFeats_stu = student_output['bev_embed']
        bevFeats_stu = bevFeats_stu.view(self.bev_resolution_stu[0], self.bev_resolution_stu[1], \
            bevFeats_stu.shape[1], bevFeats_stu.shape[2]).permute(2,3,0,1)
        if self.flag_distill_on_BEV:
            if self.bev_restore_layers is not None: # deconv for low-resolution student BEV feature
                bevFeats_stu = self.bev_restore_layers(bevFeats_stu)
            if self.norm_bevFeats is not None: # normalization for teacher BEV feature
                bevFeats_tch = self.norm_bevFeats(bevFeats_tch)
            # generate heatmap mask from GT
            heatmaps = self.teacher.pts_bbox_head.get_heatmaps(kwargs_T['gt_bboxes_3d'], kwargs_T['gt_labels_3d'])
            mask_heatmap = [torch.cat(htmap).max(dim=0).values.unsqueeze(0) for htmap in heatmaps]
            mask_heatmap = torch.stack(mask_heatmap, dim=0)
            mask_heatmap = F.interpolate(mask_heatmap, size = self.bev_resolution_tch)
        
        buffer_dict = dict(self.named_buffers())

        # distill on FPN features
        if self.flag_distill_on_FPN:
            for item_loc in self.distiller_fpn:
                student_module = 'student_' + item_loc.student_module.replace('.', '_')
                teacher_module = 'teacher_' + item_loc.teacher_module.replace('.', '_')
                student_feat = buffer_dict[student_module]
                teacher_feat = buffer_dict[teacher_module]
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    student_loss[loss_name] = self.distill_losses[loss_name](student_feat, teacher_feat)

        # distill on BEV feature
        if self.flag_distill_on_BEV:
            for item_loc in self.distiller_bev:
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    masked_bevFeats_stu = bevFeats_stu * mask_heatmap
                    masked_bevFeats_tch = bevFeats_tch * mask_heatmap
                    student_loss[loss_name] = (1/mask_heatmap.sum()) * self.distill_losses[loss_name](masked_bevFeats_stu, masked_bevFeats_tch)        
        
        
                    
        return student_loss
    
    def forward_test(self, **kwargs):
        return self.student.forward_test(**kwargs)
    
    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)

    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)