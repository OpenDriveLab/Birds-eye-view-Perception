# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS

from mmdet3d.models.detectors import BEVDepth4D


@DETECTORS.register_module()
class BEVStereo(BEVDepth4D):

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'])
        imgs, rots, trans, intrins, post_rots, post_trans, bda, sensor2sensors = \
            self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only

        for img, rot, tran, intrin, post_rot, post_tran, sensor2sensor in zip(
                imgs, rots, trans, intrins, post_rots, post_trans, sensor2sensors):
            mlp_input = self.img_view_transformer.get_mlp_input(
                rots[0][:, 0], trans[0][:, 0], intrin, post_rot, post_tran, bda)
            inputs_curr = (img, rot, tran, intrin, post_rot,
                            post_tran, bda, mlp_input, sensor2sensor)
            if key_frame:
                bev_feats, depths = self.prepare_bev_feat(*inputs_curr, **kwargs)
            else:
                with torch.no_grad():
                    bev_feats, depths = self.prepare_bev_feat(*inputs_curr, **kwargs)
            bev_feat_list.append(bev_feats)
            depth_list.append(depths)
            key_frame = False
        if self.align_after_view_transfromation:
            for adj_id in range(1, len(bev_feat_list)):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                    [trans[0][:, 0], trans[adj_id][:, 0]],
                                    [rots[0][:, 0], rots[adj_id][:, 0]],
                                    bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_list[0]

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        B, N, _, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, 3, H, W)
        imgs = torch.split(imgs, 2, 2)
        # imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda, sensor2sensors = inputs[1:8]
        extra = [
            rots.view(B, self.num_frame, N, 3, 3),
            trans.view(B, self.num_frame, N, 3),
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3),
            sensor2sensors.view(B, self.num_frame, N, 4, 4),
        ]
        extra = [torch.split(t, 2, 1) for t in extra]
        # extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans, sensor2sensors = extra
        return imgs, rots, trans, intrins, post_rots, post_trans, bda, sensor2sensors

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input, sensor2sensor, **kwargs):
        img_feats = []
        stereo_feats = []
        for idx in range(2):
            if idx == 1:
                with torch.no_grad():
                    x, stereo_feat = self.image_encoder(img[:, :, idx])
            else:
                x, stereo_feat = self.image_encoder(img[:, :, idx])
            img_feats.append(x)
            stereo_feats.append(stereo_feat)
        bev_feat, depth = self.img_view_transformer(
            [img_feats, stereo_feats, rot, tran, intrin, post_rot, post_tran, bda, mlp_input, sensor2sensor], **kwargs)
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth

    def image_encoder(self,imgs):
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x_feats = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x_feats[1:])
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        stereo_feat = x_feats[0]
        # stereo_feat = stereo_feat.view(B, N, *stereo_feat.shape[1:])
        return x, stereo_feat

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
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
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses
