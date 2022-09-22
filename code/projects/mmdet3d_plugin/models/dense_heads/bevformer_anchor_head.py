# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule, force_fp32, auto_fp16
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.cnn import ConvModule, Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, reduce_mean, build_bbox_coder)
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid

from mmdet3d.core.bbox import bbox_overlaps_nearest_3d
from mmdet3d.models.dense_heads.anchor3d_head import Anchor3DHead
from mmdet3d.models.dense_heads.train_mixins import get_direction_target


@HEADS.register_module()
class BEVFormer_FreeAnchor3DHead(Anchor3DHead):
    r"""`FreeAnchor <https://arxiv.org/abs/1909.02466>`_ head for 3D detection.

    Note:
        This implementation is directly modified from the `mmdet implementation
        <https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/free_anchor_retina_head.py>`_.
        We find it also works on 3D detection with minor modification, i.e.,
        different hyper-parameters and a additional direction classifier.

    Args:
        pre_anchor_topk (int): Number of boxes that be token in each bag.
        bbox_thr (float): The threshold of the saturated linear function. It is
            usually the same with the IoU threshold used in NMS.
        gamma (float): Gamma parameter in focal loss.
        alpha (float): Alpha parameter in focal loss.
        kwargs (dict): Other arguments are the same as those in :class:`Anchor3DHead`.
    """  # noqa: E501

    def __init__(self,
                 *args,
                 transformer=None,
                 only_encoder=False,
                 positional_encoding=None,
                 bev_h=30,
                 bev_w=30,
                 init_cfg=None,
                 pre_anchor_topk=50,
                 use_dcn=False,
                 bbox_thr=0.6,
                 gamma=2.0,
                 alpha=0.5,
                 pc_range=None,
                 **kwargs):
        # bevformer

        super().__init__(init_cfg=init_cfg, **kwargs)
        self.use_dcn = use_dcn
        self.dcn_config = dict(conv_cfg=dict(type='DCNv2_silent'),
                               norm_cfg=dict(type='BN'),
                               in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               padding=1,
                               groups=4)
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.fp16_enabled = False
        self.only_encoder = only_encoder
        # end
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10

        # only use this coder to get the direction category and residual
        self.embed_dims = self.feat_channels
        self.pc_range = pc_range
        self.real_h = self.pc_range[3] - self.pc_range[0]
        self.real_w = self.pc_range[4] - self.pc_range[1]
        self.transformer = build_transformer(transformer)
        self.positional_encoding = build_positional_encoding(positional_encoding)

        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        self.fp16_enabled = False

        self.embed_dims = self.in_channels
        self.pre_anchor_topk = pre_anchor_topk
        self.bbox_thr = bbox_thr
        self.gamma = gamma
        self.alpha = alpha

        if self.use_dcn:
            self.feature_adapt_share = ConvModule(**self.dcn_config)
            self.feature_adapt_cls = ConvModule(**self.dcn_config)
            self.feature_adapt_reg = ConvModule(**self.dcn_config)

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * self.box_code_size, 1)
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(self.feat_channels, self.num_anchors * 2, 1)

    def forward_single(self, x):
        """Forward function on a single-scale feature map.
        Args:
            x (torch.Tensor): Input features.
        Returns:
            tuple[torch.Tensor]: Contain score of each class, bbox
                regression and direction classification predictions.
        """
        if self.use_dcn:
            x = self.feature_adapt_share(x)
            cls_reat = self.feature_adapt_cls(x)
            cls_score = self.conv_cls(cls_reat)
            reg_feat = self.feature_adapt_reg(x)
            bbox_pred = self.conv_reg(reg_feat)
        else:
            cls_score = self.conv_cls(x)
            bbox_pred = self.conv_reg(x)
        dir_cls_preds = None
        if self.use_direction_classifier:
            if self.use_dcn:
                dir_cls_preds = self.conv_dir_cls(reg_feat)
            else:
                dir_cls_preds = self.conv_dir_cls(x)
        return {'all_cls_scores': cls_score, 'all_bbox_preds': bbox_pred, 'dir_cls_preds': dir_cls_preds}

    @force_fp32(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, return_bev=False, gt_bboxes_3d=None, **kwargs):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head xwith normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
            gt_bboxes_3d: for debug
        """

        # for i,each in enumerate(mlvl_feats):
        #    print('mlvl_feats',i,each.shape)
        bs, nm, c, input_img_h, input_img_w = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device
        # query_embeds = self.query_embedding.weight.to(dtype)
        bev_embeds = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        # print(bev_pos.shape,bev_embeds.shape)

        # img_masks = mlvl_feats[0].new_ones(
        #     (bs, input_img_h, input_img_w))
        # mlvl_masks = []
        # mlvl_positional_encodings = []
        # # from IPython import embed
        # # embed()
        # # exit()

        outputs = self.transformer(
            mlvl_feats,
            bev_embeds,
            None,
            self.bev_h,
            self.bev_w,
            gird_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=None,  # noqa:E501
            cls_branches=None,
            img_metas=img_metas,
            prev_bev=prev_bev,
            return_bev=return_bev,
            gt_bboxes_3d=gt_bboxes_3d,
        )
        if return_bev:
            bev_outputs, outputs = outputs  # for model save current bev feature when testing

        if self.only_encoder:
            return outputs  # for eval_model getting bev feature
        else:
            bev_feature = outputs

        bev_feature = bev_feature.permute(0, 2, 1).view(bs, self.embed_dims, self.bev_h, self.bev_w)

        outs = self.forward_single(bev_feature)

        if return_bev:
            return bev_outputs, outs
        else:
            return outs

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds'))
    def loss(
        self,
        gt_bboxes_3d,
        gt_labels_3d,
        preds_dicts,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        """Calculate loss of FreeAnchor head.

        Args:
            cls_scores (list[torch.Tensor]): Classification scores of
                different samples.
            bbox_preds (list[torch.Tensor]): Box predictions of
                different samples
            dir_cls_preds (list[torch.Tensor]): Direction predictions of
                different samples
            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]): Ground truth boxes.
            gt_labels (list[torch.Tensor]): Ground truth labels.
            input_metas (list[dict]): List of input meta information.
            gt_bboxes_ignore (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth boxes that should be ignored. Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Loss items.

                - positive_bag_loss (torch.Tensor): Loss of positive samples.
                - negative_bag_loss (torch.Tensor): Loss of negative samples.
        """
        gt_labels = gt_labels_3d
        gt_bboxes = gt_bboxes_3d
        input_metas = img_metas
        # from IPython import embed
        # embed()
        # exit()
        cls_scores = [preds_dicts['all_cls_scores']]
        bbox_preds = [preds_dicts['all_bbox_preds']]
        dir_cls_preds = [preds_dicts['dir_cls_preds']]

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        anchor_list = self.get_anchors(featmap_sizes, input_metas)
        anchors = [torch.cat(anchor) for anchor in anchor_list]

        # concatenate each level
        cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(cls_score.size(0), -1, self.num_classes) for cls_score in cls_scores
        ]
        bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(bbox_pred.size(0), -1, self.box_code_size) for bbox_pred in bbox_preds
        ]
        dir_cls_preds = [
            dir_cls_pred.permute(0, 2, 3, 1).reshape(dir_cls_pred.size(0), -1, 2) for dir_cls_pred in dir_cls_preds
        ]

        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)
        dir_cls_preds = torch.cat(dir_cls_preds, dim=1)

        cls_prob = torch.sigmoid(cls_scores)
        box_prob = []
        num_pos = 0
        positive_losses = []
        for i in range(len(anchors)):
            anchors_, gt_labels_, gt_bboxes_, cls_prob_, bbox_preds_, dir_cls_preds_ =\
            anchors[i], gt_labels[i], gt_bboxes[i], cls_prob[i], bbox_preds[i],\
            dir_cls_preds[i]
            # for _, (anchors_, gt_labels_, gt_bboxes_, cls_prob_, bbox_preds_,
            #         dir_cls_preds_) in enumerate(
            #             zip(anchors, gt_labels, gt_bboxes, cls_prob, bbox_preds,
            #                 dir_cls_preds)):

            gt_bboxes_ = gt_bboxes_.tensor.to(anchors_.device)

            with torch.no_grad():
                # box_localization: a_{j}^{loc}, shape: [j, 4]
                # from IPython import embed
                # embed()
                # exit()
                pred_boxes = self.bbox_coder.decode(anchors_, bbox_preds_)

                # object_box_iou: IoU_{ij}^{loc}, shape: [i, j]
                object_box_iou = bbox_overlaps_nearest_3d(gt_bboxes_, pred_boxes)

                # object_box_prob: P{a_{j} -> b_{i}}, shape: [i, j]
                t1 = self.bbox_thr
                t2 = object_box_iou.max(dim=1, keepdim=True).values.clamp(min=t1 + 1e-6)
                object_box_prob = ((object_box_iou - t1) / (t2 - t1)).clamp(min=0, max=1)

                # object_cls_box_prob: P{a_{j} -> b_{i}}, shape: [i, c, j]
                num_obj = gt_labels_.size(0)
                indices = torch.stack([torch.arange(num_obj).type_as(gt_labels_), gt_labels_], dim=0)

                object_cls_box_prob = torch.sparse_coo_tensor(indices, object_box_prob)

                # image_box_iou: P{a_{j} \in A_{+}}, shape: [c, j]
                """
                from "start" to "end" implement:
                image_box_iou = torch.sparse.max(object_cls_box_prob,
                                                 dim=0).t()

                """
                # start
                box_cls_prob = torch.sparse.sum(object_cls_box_prob, dim=0).to_dense()

                indices = torch.nonzero(box_cls_prob, as_tuple=False).t_()
                if indices.numel() == 0:
                    image_box_prob = torch.zeros(anchors_.size(0), self.num_classes).type_as(object_box_prob)
                else:
                    nonzero_box_prob = torch.where((gt_labels_.unsqueeze(dim=-1) == indices[0]),
                                                   object_box_prob[:, indices[1]],
                                                   torch.tensor([0]).type_as(object_box_prob)).max(dim=0).values

                    # upmap to shape [j, c]
                    image_box_prob = torch.sparse_coo_tensor(indices.flip([0]),
                                                             nonzero_box_prob,
                                                             size=(anchors_.size(0), self.num_classes)).to_dense()
                # end

                box_prob.append(image_box_prob)

            # construct bags for objects
            match_quality_matrix = bbox_overlaps_nearest_3d(gt_bboxes_, anchors_)
            _, matched = torch.topk(match_quality_matrix, self.pre_anchor_topk, dim=1, sorted=False)
            del match_quality_matrix

            # matched_cls_prob: P_{ij}^{cls}
            matched_cls_prob = torch.gather(cls_prob_[matched], 2,
                                            gt_labels_.view(-1, 1, 1).repeat(1, self.pre_anchor_topk, 1)).squeeze(2)

            # matched_box_prob: P_{ij}^{loc}
            matched_anchors = anchors_[matched]
            matched_object_targets = self.bbox_coder.encode(matched_anchors,
                                                            gt_bboxes_.unsqueeze(dim=1).expand_as(matched_anchors))

            # direction classification loss
            loss_dir = None
            if self.use_direction_classifier:
                # also calculate direction prob: P_{ij}^{dir}
                matched_dir_targets = get_direction_target(
                    matched_anchors,
                    matched_object_targets,
                    self.dir_offset,
                    # self.dir_limit_offset,
                    one_hot=False)
                loss_dir = self.loss_dir(dir_cls_preds_[matched].transpose(-2, -1),
                                         matched_dir_targets,
                                         reduction_override='none')

            # generate bbox weights
            if self.diff_rad_by_sin:
                # from IPython import embed
                # embed()
                # exit()

                bbox_preds_[matched], matched_object_targets = \
                    self.add_sin_difference(
                        bbox_preds_[matched], matched_object_targets)
            bbox_weights = matched_anchors.new_ones(matched_anchors.size())
            # Use pop is not right, check performance
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                bbox_weights = bbox_weights * bbox_weights.new_tensor(code_weight)
            loss_bbox = self.loss_bbox(bbox_preds_[matched],
                                       matched_object_targets,
                                       bbox_weights,
                                       reduction_override='none').sum(-1)

            if loss_dir is not None:
                loss_bbox += loss_dir
            matched_box_prob = torch.exp(-loss_bbox)

            # positive_losses: {-log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) )}
            num_pos += len(gt_bboxes_)
            positive_losses.append(self.positive_bag_loss(matched_cls_prob, matched_box_prob))

        positive_loss = torch.cat(positive_losses).sum() / max(1, num_pos)

        # box_prob: P{a_{j} \in A_{+}}
        box_prob = torch.stack(box_prob, dim=0)

        # negative_loss:
        # \sum_{j}{ FL((1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg})) } / n||B||
        negative_loss = self.negative_bag_loss(cls_prob, box_prob).sum() / max(1, num_pos * self.pre_anchor_topk)

        losses = {'positive_bag_loss': positive_loss, 'negative_bag_loss': negative_loss}
        return losses

    def positive_bag_loss(self, matched_cls_prob, matched_box_prob):
        """Generate positive bag loss.

        Args:
            matched_cls_prob (torch.Tensor): Classification probability
                of matched positive samples.
            matched_box_prob (torch.Tensor): Bounding box probability
                of matched positive samples.

        Returns:
            torch.Tensor: Loss of positive samples.
        """
        # bag_prob = Mean-max(matched_prob)
        matched_prob = matched_cls_prob * matched_box_prob
        weight = 1 / torch.clamp(1 - matched_prob, 1e-12, None)
        weight /= weight.sum(dim=1).unsqueeze(dim=-1)
        bag_prob = (weight * matched_prob).sum(dim=1)
        # positive_bag_loss = -self.alpha * log(bag_prob)
        bag_prob = bag_prob.clamp(0, 1)  # to avoid bug of BCE, check
        return self.alpha * F.binary_cross_entropy(bag_prob, torch.ones_like(bag_prob), reduction='none')

    def negative_bag_loss(self, cls_prob, box_prob):
        """Generate negative bag loss.

        Args:
            cls_prob (torch.Tensor): Classification probability
                of negative samples.
            box_prob (torch.Tensor): Bounding box probability
                of negative samples.

        Returns:
            torch.Tensor: Loss of negative samples.
        """
        prob = cls_prob * (1 - box_prob)
        prob = prob.clamp(0, 1)  # to avoid bug of BCE, check
        negative_bag_loss = prob**self.gamma * F.binary_cross_entropy(prob, torch.zeros_like(prob), reduction='none')
        return (1 - self.alpha) * negative_bag_loss

    def get_bboxes(self, preds_dicts, img_metas, cfg=None, rescale=False):
        """Get bboxes of anchor head.
        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            input_metas (list[dict]): Contain pcd and img's meta info.
            cfg (:obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): Whether th rescale bbox.
        Returns:
            list[tuple]: Prediction resultes of batches.
        """
        input_metas = img_metas
        # from IPython import embed
        # embed()
        # exit()
        cls_scores = [preds_dicts['all_cls_scores']]
        bbox_preds = [preds_dicts['all_bbox_preds']]
        dir_cls_preds = [preds_dicts['dir_cls_preds']]

        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = cls_scores[0].device
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device=device)
        mlvl_anchors = [anchor.reshape(-1, self.box_code_size) for anchor in mlvl_anchors]

        result_list = []
        for img_id in range(len(input_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            dir_cls_pred_list = [dir_cls_preds[i][img_id].detach() for i in range(num_levels)]

            input_meta = input_metas[img_id]
            bboxes, scores, labels = self.get_bboxes_single(cls_score_list, bbox_pred_list, dir_cls_pred_list,
                                                            mlvl_anchors, input_meta, cfg, rescale)
            if img_metas[img_id]['flip']:
                bboxes.tensor[:, 1] = -bboxes.tensor[:, 1]
                bboxes.tensor[:, -1] = -bboxes.tensor[:, -1] + np.pi
            proposals = bboxes, scores, labels
            result_list.append(proposals)
        return result_list