# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import force_fp32
from torch.cuda.amp.autocast_mode import autocast

from mmdet.models.backbones.resnet import BasicBlock
from ..builder import NECKS

from mmdet3d.models.necks.view_transformer import LSSViewTransformer, ASPP, Mlp, SELayer
from scipy.special import erf
from scipy.stats import norm
import numpy as np
import math


class ConvBnReLU3D(nn.Module):
    """Implements of 3d convolution + batch normalization + ReLU."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
    ) -> None:
        """initialization method for convolution3D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=pad,
                              dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class DepthNet(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 d_bound=[1.0, 60.0, 0.5],
                 num_ranges=4):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_feat_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
        )
        self.mono_depth_net = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

        # TODO: bevstereo
        self.mu_sigma_range_net = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            nn.ConvTranspose2d(mid_channels,
                            mid_channels,
                            3,
                            stride=2,
                            padding=1,
                            output_padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels,
                            mid_channels,
                            3,
                            stride=2,
                            padding=1,
                            output_padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                    num_ranges * 3,
                    kernel_size=1,
                    stride=1,
                    padding=0),
        )
        self.num_ranges = num_ranges
        self.dbound = d_bound

    def forward(self, x, mlp_input):
        B, _, H, W = x.shape

        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth_feat = self.depth_se(x, depth_se)
        depth_feat = self.depth_feat_conv(depth_feat)

        mono_depth = self.mono_depth_net(depth_feat)
        
        # TODO: stereo
        mu_sigma_score = self.mu_sigma_range_net(depth_feat) 
        d_coords = torch.arange(*self.dbound,
                                dtype=torch.float).reshape(1, -1, 1, 1).cuda()
        d_coords = d_coords.repeat(B, 1, H, W)
        mu = mu_sigma_score[:, 0:self.num_ranges, ...]
        sigma = mu_sigma_score[:, self.num_ranges:2 * self.num_ranges, ...]
        range_score = mu_sigma_score[:,
                                    2 * self.num_ranges:3 * self.num_ranges,
                                    ...]
        sigma = F.elu(sigma) + 1.0 + 1e-10
        return context, mu, sigma, range_score, mono_depth

@NECKS.register_module()
class LSSViewTransformerBEVStereo(LSSViewTransformer):

    def __init__(self, loss_depth_weight=3.0, depthnet_cfg=dict(), num_groups=8, k_list=None, **kwargs):
        super(LSSViewTransformerBEVStereo, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNet(self.in_channels, self.in_channels,
                                  self.out_channels, self.D, **depthnet_cfg)
        self.stereo_downsample_factor = 4
        self.use_mask = True
        self.num_ranges = 4
        self.range_list = [[2, 8], [8, 16], [16, 28], [28, 58]]
        self.em_iteration = 3
        self.num_samples = 3
        self.min_sigma = 1
        self.sampling_range = 3
        self.num_groups = num_groups
        if k_list is None:
            self.register_buffer('k_list', torch.Tensor(self.depth_sampling()))
        else:
            self.register_buffer('k_list', torch.Tensor(k_list))
            
        self.similarity_net = nn.Sequential(
            ConvBnReLU3D(in_channels=num_groups,
                        out_channels=16,
                        kernel_size=1,
                        stride=1,
                        pad=0),
            ConvBnReLU3D(in_channels=16,
                        out_channels=8,
                        kernel_size=1,
                        stride=1,
                        pad=0),
            nn.Conv3d(in_channels=8,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0),
        )

        self.depth_downsample_net = nn.Sequential(
            nn.Conv2d(self.D, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.D, 1, 1, 0),
        )
        if self.use_mask:
            self.mask_net = nn.Sequential(
                nn.Conv2d(236, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                BasicBlock(64, 64),
                BasicBlock(64, 64),
                nn.Conv2d(64, 1, 1, 1, 0),
                nn.Sigmoid(),
            )

    def depth_sampling(self):
        """Generate sampling range of candidates.
        Returns:
            list[float]: List of all candidates.
        """
        P_total = erf(self.sampling_range /
                      np.sqrt(2))  # Probability covered by the sampling range
        idx_list = np.arange(0, self.num_samples + 1)
        p_list = (1 - P_total) / 2 + ((idx_list / self.num_samples) * P_total)
        k_list = norm.ppf(p_list)
        k_list = (k_list[1:] + k_list[:-1]) / 2
        return list(k_list)

    def _generate_cost_volume(
        self,
        stereo_feats_all_sweeps,
        mats_dict,
        depth_sample,
        depth_sample_frustum,
        sensor2sensor_mats,
    ):
        """Generate cost volume based on depth sample.
        Args:
            sweep_index (int): Index of sweep.
            stereo_feats_all_sweeps (list[Tensor]): Stereo feature
                of all sweeps.
            mats_dict (dict):
                sensor2ego_mats (Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats (Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats (Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats (Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat (Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            depth_sample (Tensor): Depth map of all candidates.
            depth_sample_frustum (Tensor): Pre-generated frustum.
            sensor2sensor_mats (Tensor): Transformation matrix from reference
                sensor to source sensor.
        Returns:
            Tensor: Depth score for all sweeps.
        """
        batch_size, num_channels, height, width = stereo_feats_all_sweeps[
            0].shape
        # thres = int(self.mvs_weighting.split("CW")[1])
        depth_score_all_sweeps = list()

        warped_stereo_fea = self.homo_warping(
            stereo_feats_all_sweeps[1],
            mats_dict['intrin_mats'][0],
            mats_dict['intrin_mats'][1],
            sensor2sensor_mats,
            mats_dict['ida_mats'][0],
            mats_dict['ida_mats'][1],
            depth_sample,
            depth_sample_frustum.type_as(stereo_feats_all_sweeps[1]),
        )
        warped_stereo_fea = warped_stereo_fea.reshape(
            batch_size, self.num_groups, num_channels // self.num_groups,
            self.num_samples, height, width)
        ref_stereo_feat = stereo_feats_all_sweeps[0].reshape(
            batch_size, self.num_groups, num_channels // self.num_groups,
            height, width)
        feat_cost = torch.mean(
            (ref_stereo_feat.unsqueeze(3) * warped_stereo_fea), axis=2)
        depth_score = self.similarity_net(feat_cost).squeeze(1)
        depth_score_all_sweeps.append(depth_score)
        return torch.stack(depth_score_all_sweeps).mean(0)

    def homo_warping(
        self,
        stereo_feat,
        key_intrin_mats,
        sweep_intrin_mats,
        sensor2sensor_mats,
        key_ida_mats,
        sweep_ida_mats,
        depth_sample,
        frustum,
    ):
        """Used for mvs method to transfer sweep image feature to
            key image feature.
        Args:
            src_fea(Tensor): image features.
            key_intrin_mats(Tensor): Intrin matrix for key sensor.
            sweep_intrin_mats(Tensor): Intrin matrix for sweep sensor.
            sensor2sensor_mats(Tensor): Transformation matrix from key
                sensor to sweep sensor.
            key_ida_mats(Tensor): Ida matrix for key frame.
            sweep_ida_mats(Tensor): Ida matrix for sweep frame.
            depth_sample (Tensor): Depth map of all candidates.
            depth_sample_frustum (Tensor): Pre-generated frustum.
        """
        batch_size_with_num_cams, channels = stereo_feat.shape[
            0], stereo_feat.shape[1]
        height, width = stereo_feat.shape[2], stereo_feat.shape[3]
        with torch.no_grad():
            points = frustum
            points = points.reshape(points.shape[0], -1, points.shape[-1])
            points[..., 2] = 1
            # Undo ida for key frame.
            points = key_ida_mats.reshape(batch_size_with_num_cams,
                                          *key_ida_mats.shape[2:]).inverse(
                                          ).unsqueeze(1) @ points.unsqueeze(-1)
            # Convert points from pixel coord to key camera coord.
            points[..., :3, :] *= depth_sample.reshape(batch_size_with_num_cams, -1, 1, 1) 
            num_depth = frustum.shape[1]
            points = (key_intrin_mats.reshape(
                batch_size_with_num_cams,
                *key_intrin_mats.shape[2:]).inverse().unsqueeze(1) @ points)

            points = (sensor2sensor_mats.reshape(
                batch_size_with_num_cams,
                *sensor2sensor_mats.shape[2:]).unsqueeze(1) @ points)
            # points in sweep sensor coord.
            points = (sweep_intrin_mats.reshape(
                batch_size_with_num_cams,
                *sweep_intrin_mats.shape[2:]).unsqueeze(1) @ points)
            # points in sweep pixel coord.
            points[..., :2, :] = points[..., :2, :] / points[
                ..., 2:3, :]  # [B, 2, Ndepth, H*W]

            points = (sweep_ida_mats.reshape(
                batch_size_with_num_cams,
                *sweep_ida_mats.shape[2:]).unsqueeze(1) @ points).squeeze(-1)
            neg_mask = points[..., 2] < 1e-3
            points[..., 0][neg_mask] = width * self.stereo_downsample_factor
            points[..., 1][neg_mask] = height * self.stereo_downsample_factor
            points[..., 2][neg_mask] = 1
            proj_x_normalized = points[..., 0] / (
                (width * self.stereo_downsample_factor - 1) / 2) - 1
            proj_y_normalized = points[..., 1] / (
                (height * self.stereo_downsample_factor - 1) / 2) - 1
            grid = torch.stack([proj_x_normalized, proj_y_normalized],
                               dim=2)  # [B, Ndepth, H*W, 2]

        warped_stereo_fea = F.grid_sample(
            stereo_feat,
            grid.view(batch_size_with_num_cams, num_depth * height, width, 2),
            mode='bilinear',
            padding_mode='zeros',
        )
        warped_stereo_fea = warped_stereo_fea.view(batch_size_with_num_cams,
                                                   channels, num_depth, height,
                                                   width)

        return warped_stereo_fea

    def _forward_stereo(
        self,
        stereo_feats_all_sweeps,
        mono_depth_all_sweeps,
        mats_dict,
        sensor2sensor_mats,
        mu_all_sweeps,
        sigma_all_sweeps,
        range_score_all_sweeps,
    ):
        """Forward function to generate stereo depth.
        Args:
            sweep_index (int): Index of sweep.
            stereo_feats_all_sweeps (list[Tensor]): Stereo feature
                of all sweeps.
            mono_depth_all_sweeps (list[Tensor]):
            mats_dict (dict):
                sensor2ego_mats (Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats (Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats (Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats (Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat (Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            sensor2sensor_mats(Tensor): Transformation matrix from key
                sensor to sweep sensor.
            mu_all_sweeps (list[Tensor]): List of mu for all sweeps.
            sigma_all_sweeps (list[Tensor]): List of sigma for all sweeps.
            range_score_all_sweeps (list[Tensor]): List of all range score
                for all sweeps.
            depth_feat_all_sweeps (list[Tensor]): List of all depth feat for
                all sweeps.
        Returns:
            Tensor: stereo_depth
        """
        batch_size_with_cams, _, feat_height, feat_width = \
            stereo_feats_all_sweeps[0].shape
        device = stereo_feats_all_sweeps[0].device
        d_coords = torch.arange(*self.grid_config['depth'],
                                dtype=torch.float,
                                device=device).reshape(1, -1, 1, 1)
        d_coords = d_coords.repeat(batch_size_with_cams, 1, feat_height,
                                   feat_width)
        stereo_depth = stereo_feats_all_sweeps[0].new_zeros(
            batch_size_with_cams, self.D, feat_height, feat_width)
        mask_score = stereo_feats_all_sweeps[0].new_zeros(
            batch_size_with_cams,
            self.D,
            feat_height * self.stereo_downsample_factor //
            self.downsample,
            feat_width * self.stereo_downsample_factor //
            self.downsample,
        )
        score_all_ranges = list()
        range_score = range_score_all_sweeps[0].softmax(1)
        for range_idx in range(self.num_ranges):
            # Map mu to the corresponding interval.
            range_start = self.range_list[range_idx][0]
            mu_all_sweeps_single_range = [
                mu[:, range_idx:range_idx + 1, ...].sigmoid() *
                (self.range_list[range_idx][1] - self.range_list[range_idx][0])
                + range_start for mu in mu_all_sweeps
            ]
            sigma_all_sweeps_single_range = [
                sigma[:, range_idx:range_idx + 1, ...]
                for sigma in sigma_all_sweeps
            ]
            batch_size_with_cams, _, feat_height, feat_width =\
                stereo_feats_all_sweeps[0].shape
            mu = mu_all_sweeps_single_range[0]
            sigma = sigma_all_sweeps_single_range[0]
            for _ in range(self.em_iteration):
                depth_sample = torch.cat([mu + sigma * k for k in self.k_list],
                                         1)
                depth_sample_frustum = self.create_depth_sample_frustum(
                    depth_sample, self.stereo_downsample_factor)
                mu_score = self._generate_cost_volume(
                    stereo_feats_all_sweeps,
                    mats_dict,
                    depth_sample,
                    depth_sample_frustum,
                    sensor2sensor_mats,
                )
                mu_score = mu_score.softmax(1)
                scale_factor = torch.clamp(
                    0.5 / (1e-4 + mu_score[:, self.num_samples //
                                           2:self.num_samples // 2 + 1, ...]),
                    min=0.1,
                    max=10)

                sigma = torch.clamp(sigma * scale_factor, min=0.1, max=10)
                mu = (depth_sample * mu_score).sum(1, keepdim=True)
                del depth_sample
                del depth_sample_frustum
            mu = torch.clamp(mu,
                             max=self.range_list[range_idx][1],
                             min=self.range_list[range_idx][0])
            range_length = int(
                (self.range_list[range_idx][1] - self.range_list[range_idx][0])
                // self.grid_config['depth'][2])
            if self.use_mask:
                depth_sample = F.avg_pool2d(
                    mu,
                    self.downsample // self.stereo_downsample_factor,
                    self.downsample // self.stereo_downsample_factor,
                )
                depth_sample_frustum = self.create_depth_sample_frustum(
                    depth_sample, self.downsample)
                mask = self._forward_mask(
                    mono_depth_all_sweeps,
                    mats_dict,
                    depth_sample,
                    depth_sample_frustum,
                    sensor2sensor_mats,
                )
                mask_score[:,
                           int((range_start - self.grid_config['depth'][0]) //
                               self.grid_config['depth'][2]):range_length +
                           int((range_start - self.grid_config['depth'][0]) //
                               self.grid_config['depth'][2]), ..., ] += mask
                del depth_sample
                del depth_sample_frustum
            sigma = torch.clamp(sigma, self.min_sigma)
            mu_repeated = mu.repeat(1, range_length, 1, 1)
            eps = 1e-6
            depth_score_single_range = (-1 / 2 * (
                (d_coords[:,
                          int((range_start - self.grid_config['depth'][0]) //
                              self.grid_config['depth'][2]):range_length + int(
                                  (range_start - self.grid_config['depth'][0]) //
                                  self.grid_config['depth'][2]), ..., ] - mu_repeated) /
                torch.sqrt(sigma))**2)
            depth_score_single_range = depth_score_single_range.exp()
            score_all_ranges.append(mu_score.sum(1).unsqueeze(1))
            depth_score_single_range = depth_score_single_range / (
                sigma * math.sqrt(2 * math.pi) + eps)
            stereo_depth[:,
                         int((range_start - self.grid_config['depth'][0]) //
                             self.grid_config['depth'][2]):range_length +
                         int((range_start - self.grid_config['depth'][0]) //
                             self.grid_config['depth'][2]), ..., ] = (
                                 depth_score_single_range *
                                 range_score[:, range_idx:range_idx + 1, ...])
            # del range_score
            del depth_score_single_range
            del mu_repeated
        if self.use_mask:
            return stereo_depth, mask_score
        else:
            return stereo_depth

    def create_depth_sample_frustum(self, depth_sample, downsample_factor=16):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.input_size
        fH, fW = ogfH // downsample_factor, ogfW // downsample_factor
        batch_size, num_depth, _, _ = depth_sample.shape
        x_coords = (torch.linspace(0,
                                   ogfW - 1,
                                   fW,
                                   dtype=torch.float,
                                   device=depth_sample.device).view(
                                       1, 1, 1,
                                       fW).expand(batch_size, num_depth, fH,
                                                  fW))
        y_coords = (torch.linspace(0,
                                   ogfH - 1,
                                   fH,
                                   dtype=torch.float,
                                   device=depth_sample.device).view(
                                       1, 1, fH,
                                       1).expand(batch_size, num_depth, fH,
                                                 fW))
        paddings = torch.ones_like(depth_sample)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, depth_sample, paddings), -1)
        return frustum

    def _forward_mask(
        self,
        mono_depth_all_sweeps,
        mats_dict,
        depth_sample,
        depth_sample_frustum,
        sensor2sensor_mats,
    ):
        """Forward function to generate mask.
        Args:
            sweep_index (int): Index of sweep.
            mono_depth_all_sweeps (list[Tensor]): List of mono_depth for
                all sweeps.
            mats_dict (dict):
                sensor2ego_mats (Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats (Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats (Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats (Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat (Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            depth_sample (Tensor): Depth map of all candidates.
            depth_sample_frustum (Tensor): Pre-generated frustum.
            sensor2sensor_mats (Tensor): Transformation matrix from reference
                sensor to source sensor.
        Returns:
            Tensor: Generated mask.
        """
        mask_all_sweeps = list()

        warped_mono_depth = self.homo_warping(
            mono_depth_all_sweeps[1],
            mats_dict['intrin_mats'][0],
            mats_dict['intrin_mats'][1],
            sensor2sensor_mats,
            mats_dict['ida_mats'][0],
            mats_dict['ida_mats'][1],
            depth_sample,
            depth_sample_frustum.type_as(mono_depth_all_sweeps[1]),
        )
        mask = self.mask_net(
            torch.cat([
                mono_depth_all_sweeps[0].detach(),
                warped_mono_depth.mean(2).detach()
            ], 1))
        mask_all_sweeps.append(mask)
        return torch.stack(mask_all_sweeps).mean(0)


    def get_mlp_input(self, rot, tran, intrins, post_rots, post_trans, bda):
        mlp_inputs = []
        B, N, _, _ = rot.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        for i in range(2):
            mlp_input = torch.stack([
                intrins[:, i][:, :, 0, 0],
                intrins[:, i][:, :, 1, 1],
                intrins[:, i][:, :, 0, 2],
                intrins[:, i][:, :, 1, 2],
                post_rots[:, i][:, :, 0, 0],
                post_rots[:, i][:, :, 0, 1],
                post_trans[:, i][:, :, 0],
                post_rots[:, i][:, :, 1, 0],
                post_rots[:, i][:, :, 1, 1],
                post_trans[:, i][:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],
            ],
                                    dim=-1)
            sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)],
                                dim=-1).reshape(B, N, -1)
            mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
            mlp_inputs.append(mlp_input)
        
        return mlp_inputs

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)

        gt_depths = (
            gt_depths -
            (self.grid_config['depth'][0] -
             self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss

    # def voxelization(self, depth_probs, rots, trans, cam2imgs, post_rots, post_trans,):
    #     B, N, D, H, W= depth_probs.shape
    #     W_in = 128
    #     H_in = 128
    #     d = torch.arange(*self.grid_config['depth'], dtype=torch.float)\
    #         .view(-1, 1, 1).expand(-1, H, W)
    #     self.D = d.shape[0]
    #     x = torch.linspace(0, W_in - 1, W,  dtype=torch.float)\
    #         .view(1, 1, W).expand(self.D, H, W)
    #     y = torch.linspace(0, H_in - 1, H,  dtype=torch.float)\
    #         .view(1, H, 1).expand(self.D, H, W)
        
    #     # D x H x W x 3
    #     self.frustum = torch.stack((x, y, d), -1)

    #     points = self.frustum.to(rots) - post_trans.view(B, N, 1, 1, 1, 3)
    #     points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
    #         .matmul(points.unsqueeze(-1))

    #     # cam_to_ego
    #     points = torch.cat(
    #         (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
    #     combine = rots.matmul(torch.inverse(cam2imgs))
    #     points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
    #     points += trans.view(B, N, 1, 1, 1, 3)
    #     return points

    def forward(self, input,  **kwargs):
        (x_feats, stereo_feats, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_inputs, sensor2sensors) = input[:10]
        B, N, C, H, W = x_feats[0].shape

        # TODO: making mats_dict
        mats_dict = {}
        intrin_mats = []
        ida_mats = []
        for i in range(2):
            intrin_mat = torch.zeros(B, N, 4, 4).to(x_feats[0].device)
            intrin_mat[:, :, 3, 3] = 1
            intrin_mat[:, :, :3, :3] = intrins[:, i]

            ida_mat = torch.zeros(B, N, 4, 4).to(x_feats[0].device)
            ida_mat[:, :, 3, 3] = 1
            ida_mat[:, :, 2, 2] = 1
            ida_mat[:, :, :2, :2] = post_rots[:, i][:, :, :2, :2]
            ida_mat[:, :, :2, 3] = post_trans[:, i][:, :, :2]

            intrin_mats.append(intrin_mat)
            ida_mats.append(ida_mat)

        mats_dict['intrin_mats'] = intrin_mats
        mats_dict['ida_mats'] = ida_mats
        
        mu_all = list()
        sigma_all = list()
        range_score_all = list()
        mono_depth_all = list()
        context_all = list()

        for idx in range(2):
            if idx == 1:
                with torch.no_grad():
                    x_feat = x_feats[idx]
                    x_feat = x_feat.view(B * N, C, H, W)
                    context, mu, sigma, range_score, mono_depth = self.depth_net(x_feat, mlp_inputs[idx])
            else:
                x_feat = x_feats[idx]
                x_feat = x_feat.view(B * N, C, H, W)
                context, mu, sigma, range_score, mono_depth = self.depth_net(x_feat, mlp_inputs[idx])
            
            mu_all.append(mu)
            sigma_all.append(sigma)
            range_score_all.append(range_score)
            mono_depth_all.append(mono_depth)
            context_all.append(context)
                
        ref2keysensor_mats = sensor2sensors[:, 0].inverse()
        key2srcsensor_mats = sensor2sensors[:, 1]
        ref2srcsensor_mats = key2srcsensor_mats @ ref2keysensor_mats
        stereo_depth, mask = self._forward_stereo(
            stereo_feats,
            mono_depth_all,
            mats_dict,
            ref2srcsensor_mats,
            mu_all,
            sigma_all,
            range_score_all,
            )
    
        depth_score = (
            mono_depth_all[0] +
            self.depth_downsample_net(stereo_depth) * mask)
        depth_prob_loss_input = depth_score.softmax(dim=1) # torch.Size([6, 118, 16, 44])
        depth_prob = depth_prob_loss_input.view(B, N, self.D, H, W)

        depth_gt = kwargs['gt_depth'] # torch.Size([1, 6, 256, 704])
        B_depth_gt, N_frame_cam, H_depth_gt, W_depth_gt = depth_gt.shape # N_depth_gt = N_cam(6) * N_frames(2)
        #flag_depth_gts = torch.split(depth_gt, 1, dim=1)
        #flag_depth_gts = [gt.reshape(B_depth_gt * N_frame_cam//2, 1, H_depth_gt, W_depth_gt) for gt in flag_depth_gts]
        depth_gt = F.interpolate(depth_gt, size=(H, W), mode='bilinear', align_corners=False)
        flag_depth_gts = depth_gt.unsqueeze(2)
        depth_gt = (depth_gt - self.grid_config['depth'][0])\
            /self.grid_config['depth'][2]
        depth_gt = torch.clip(torch.floor(depth_gt), 0, self.D).to(torch.long)
        depth_gt_logits = F.one_hot(depth_gt, num_classes=self.D) # torch.Size([1, 6, 256, 704, 118])
        depth_gt_logits = depth_gt_logits.permute(0, 1, 4, 2, 3).to(torch.float32) # (bs, N_frame, N_cam, D, H, W)
        #depth_gt_logits = torch.split(depth_gt_logits, 1, dim=1)
        # depth_gt_logits = [logit.squeeze(1).reshape(B_depth_gt * N_frame_cam//2, 
        #         self.D, H_depth_gt, W_depth_gt) for logit in depth_gt_logits] # [torch.Size([6, 59, 16, 44])] * n_frame
        depth_fusion = (flag_depth_gts!=0).int() *  depth_gt_logits + \
                 (flag_depth_gts==0).int() * depth_prob
        # for idx in range(len(mono_depth_all)):
        #     mono_depth_all_oracle.append((flag_depth_gts[idx]!=0).int() *  depth_gt_logits[idx] + \
        #         (flag_depth_gts[idx]==0).int() * mono_depth_all[idx])

        new_input = [x_feats[0], rots[:, 0], trans[:, 0], intrins[:, 0], post_rots[:, 0], post_trans[:, 0], bda]

        bev_feat, depth_fusion = self.view_transform(new_input, depth_fusion, context_all[0])
        # points = self.voxelization(depth_fusion, rots, trans, intrins, post_rots, post_trans)
        return bev_feat, depth_prob_loss_input
