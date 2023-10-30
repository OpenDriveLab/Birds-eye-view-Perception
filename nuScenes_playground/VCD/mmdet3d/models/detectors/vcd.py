# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
import torch.nn as nn
import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt
import cv2

def convert_color(img_path):
    plt.figure()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.imsave(img_path, img, cmap=plt.get_cmap('viridis'))
    plt.close()


def save_tensor(tensor, path, pad_value=254.0,normalize=False):
    print('save_tensor', path)
    tensor = tensor.to(torch.float).detach().cpu()
    max_ = tensor.flatten(1).max(-1).values[:, None, None]
    min_ = tensor.flatten(1).min(-1).values[:, None, None]
    tensor = (tensor-min_)/(max_-min_)
    if tensor.type() == 'torch.BoolTensor':
        tensor = tensor*255
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    tensor = make_grid(tensor, pad_value=pad_value, normalize=normalize).permute(1, 2, 0).numpy().copy()
    torchvision.utils.save_image(torch.tensor(tensor).permute(2, 0, 1), path)
    convert_color(path)


def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res

@DETECTORS.register_module()
class VCD(CenterPoint):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self, img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck,
                 key_frames=0,
                 pre_process=None,
                # soloffusion
                 do_history=True,
                 interpolation_mode='bilinear',
                 history_cat_num=16,
                 history_cat_conv_out_channels=None,
                 single_bev_num_channels=80,
                **kwargs):
        
        self.use_depth_gt = False
        self.use_depth_gt_only = False
        self.use_weighted_gt = False
        if 'use_depth_gt' in kwargs:
            self.use_depth_gt = kwargs['use_depth_gt']
            kwargs.pop('use_depth_gt')
        if 'use_depth_gt_only' in kwargs:
            self.use_depth_gt_only = kwargs['use_depth_gt_only']
            kwargs.pop('use_depth_gt_only')
        if 'use_weighted_gt' in kwargs:
            self.use_weighted_gt = kwargs['use_weighted_gt']
            kwargs.pop('use_weighted_gt')

        super(VCD, self).__init__(**kwargs)

        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)


        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = \
            builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

        #### Deal with history
        self.single_bev_num_channels = single_bev_num_channels
        self.do_history = do_history
        if True:
            self.interpolation_mode = interpolation_mode

            self.history_cat_num = history_cat_num
            self.history_cam_sweep_freq = 0.5 # seconds between each frame
            history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                             if history_cat_conv_out_channels is not None 
                                             else self.single_bev_num_channels)
            # Embed each sample with its relative temporal offset with current timestep
            # conv = nn.Conv2d if self.img_view_transformer.nx[-1] == 1 else nn.Conv3d
            # self.history_keyframe_time_conv = nn.Sequential(
            #      conv(self.single_bev_num_channels + 1,
            #              self.single_bev_num_channels,
            #              kernel_size=1,
            #              padding=0,
            #              stride=1),
            #      nn.SyncBatchNorm(self.single_bev_num_channels),
            #      nn.ReLU(inplace=True))

            # # Then concatenate and send them through an MLP.
            # self.history_keyframe_cat_conv = nn.Sequential(
            #     conv(self.single_bev_num_channels * (self.history_cat_num + 1),
            #             history_cat_conv_out_channels,
            #             kernel_size=1,
            #             padding=0,
            #             stride=1),
            #     nn.SyncBatchNorm(history_cat_conv_out_channels),
            #     nn.ReLU(inplace=True))

            self.history_sweep_time = None

            self.history_bev = None
            self.history_seq_ids = None
            self.history_forward_augs = None
            self.count = 0
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    @force_fp32()
    def fuse_history(self, curr_bev, img_metas, bda):
        '''
        '''
        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        if voxel_feat:
            curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
        
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)


        # print('sqe_ids', seq_ids, ' start_of_sequence ', start_of_sequence.item(), ' index ', img_metas[0]['index'])


        forward_augs = generate_forward_transformation_matrix(bda)
            # torch.stack([
            #     generate_forward_transformation_matrix(single_img_metas) 
            #     for single_img_metas in img_metas], dim=0).to(curr_bev)
        
        curr_to_prev_ego_rt = torch.stack([
            single_img_metas['curr_to_prev_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)

        ## Deal with first batch

        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            if voxel_feat:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1) 
            else:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)
        self.history_bev = self.history_bev.detach()

        assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.

        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
                "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)

        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_sweep_time += 1 # new timestep, everything in history gets pushed back one.
        if start_of_sequence.sum()>0:
            if voxel_feat:    
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1)
            else:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
            
            self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]


        ## Get grid idxs & grid2bev first.
        if voxel_feat:
            n, c_, z, h, w = curr_bev.shape
            curr_bev = curr_bev.view(n, c_*z, h, w)

        n, c, h, w = curr_bev.shape

        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1).expand(h, w)
        grid = torch.stack(
            (xs, ys, torch.ones_like(xs), torch.ones_like(xs)), -1).view(1, h, w, 4).expand(n, h, w, 4).view(n,h,w,4,1)

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.dx[0]
        feat2bev[1, 1] = self.img_view_transformer.dx[1]
        feat2bev[0, 3] = self.img_view_transformer.bx[0] - self.img_view_transformer.dx[0] / 2.
        feat2bev[1, 3] = self.img_view_transformer.bx[1] - self.img_view_transformer.dx[1] / 2.
        feat2bev[2, 2] = 1
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)

        ## Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations.
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                   @ torch.inverse(forward_augs) @ feat2bev)

        grid = rt_flow.view(n, 1, 1, 4, 4) @ grid

        # normalize and sample
        normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid = grid[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        tmp_bev = self.history_bev
        if voxel_feat: 
            n, mc, z, h, w = tmp_bev.shape
            tmp_bev = tmp_bev.reshape(n, mc * z, h, w)
        sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype), align_corners=True, mode=self.interpolation_mode)




        ## Update history
        # Add in current frame to features & timestep
        # self.history_sweep_time = torch.cat(
        #     [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
        #     dim=1) # B x (1 + T)

        if voxel_feat:
            sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
            curr_bev = curr_bev.reshape(n, c_, z, h, w)
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W 


        # Reshape and concatenate features and timestep

        feats_to_return = feats_cat.reshape(
                feats_cat.shape[0], (self.history_cat_num + 1) * self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W


        # vis_feat = feats_to_return.reshape(n, 9, -1, h, w) 
        # vis_feat = vis_feat[0,:,:]
        # vis_feat = vis_feat.permute(0,2,3,1).abs().mean(-1)
        # #[0, 0, ::16, 4]
        # # vis_feat_2 = curr_bev.reshape(n, 80, z, h, w)[0, ::16, 4]
        # save_tensor(vis_feat, '%s.png' % str(img_metas[0]['index']))
        # save_tensor(vis_feat_2, '%s.png' % str(img_metas[0]['index']))

        # if voxel_feat:
        #     feats_to_return = torch.cat(
        #     [feats_to_return, self.history_sweep_time[:, :, None, None, None, None].repeat(
        #         1, 1, 1, *feats_to_return.shape[3:]) * self.history_cam_sweep_freq
        #     ], dim=2) # B x (1 + T) x 81 x Z x H x W
        # else:
        #     feats_to_return = torch.cat(
        #     [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
        #         1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
        #     ], dim=2) # B x (1 + T) x 81 x H x W

        # Time conv
        # feats_to_return = self.history_keyframe_time_conv(
        #     feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
        #         feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 [xZ] x H x W


        # # Cat keyframes & conv
        # feats_to_return = self.history_keyframe_cat_conv(
        #     feats_to_return.reshape(
        #         feats_to_return.shape[0], -1, *feats_to_return.shape[3:])) # B x C x H x W or B x C x Z x H x W
        
        # self.down2x = nn.MaxPool3d(2, 2, 0)
        # gt_occupancy[gt_occupancy==255] = 0
        # gt_occupancy = self.down2x(gt_occupancy.to(torch.float)).to(torch.int)
        
        # if  self.count > 0:
        #     from IPython import embed
        #     embed()
        #     exit()
        # self.count += 1
        # self.history_gt = gt_occupancy
        # Update history by moving everything down one group of single_bev_num_channels channels
        # and adding in curr_bev.
        # Clone is necessary since we're doing in-place operations on self.history_bev
        # history_gt = F.grid_sample(self.history_gt.permute(0, 3, 1, 2).to(torch.float), grid.to(curr_bev.dtype), align_corners=True, mode='nearest')
        # history_gt = history_gt.permute(0, 2, 3, 1)

        self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        if voxel_feat:
            feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        return feats_to_return.clone()

    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x


    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        img_fpn_feats = self.image_encoder(img[0])
        kwargs['use_depth_gt'] = self.use_depth_gt
        kwargs['use_depth_gt_only'] = self.use_depth_gt_only
        kwargs['use_weighted_gt'] = self.use_weighted_gt
        kwargs['sweep_idx'] = 0
        mlp_input = self.img_view_transformer.get_mlp_input(*img[1:7])
        x, depth, depth_fusion = self.img_view_transformer([img_fpn_feats] + img[1:7] + [mlp_input], **kwargs)
        # Fuse History
        if self.pre_process:
            x = self.pre_process_net(x)[0]
        x = self.fuse_history(x, img_metas, img[6])
        
        x = self.bev_encoder(x)
        
        if kwargs['flag_distill']:
            return [x], depth, depth_fusion, img_fpn_feats
        else:
            return [x], depth

    def extract_feat(self, points, img, img_metas, flag_distill = False, **kwargs):
        """Extract features from images and points."""
        kwargs['flag_distill'] = flag_distill
        if flag_distill:
            bev_feats, depth, depth_fusion, img_fpn_feats  = self.extract_img_feat(img, img_metas, **kwargs)
            pts_feats = None
            return (bev_feats, pts_feats, depth, depth_fusion, img_fpn_feats)
        else:
            bev_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)
            pts_feats = None
            return (bev_feats, pts_feats, depth)

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
                      # gt_depth=None,
                      flag_distill=False,
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
        if flag_distill is False:
            bev_feats, pts_feats, depth = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas, flag_distill = flag_distill, **kwargs)
        else:
            bev_feats, pts_feats, depth, depth_fusion, img_fpn_feats = self.extract_feat(
                points, img=img_inputs, img_metas=img_metas, flag_distill = flag_distill, **kwargs)
        gt_depth = kwargs['gt_depth'][:,:6,:,:]
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(bev_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        if not flag_distill:
            return losses
        else:
            #return losses, img_feats_to_distill, bev_feats_to_distill, img_feats[0]
            return losses, bev_feats[0], img_fpn_feats, depth, depth_fusion

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        self.do_history = True
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs



