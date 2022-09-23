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


import numpy as np
import torch
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVTransformerEncoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_pre_bev_layer=3, return_intermediate=False, **kwargs):

        super(BEVTransformerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.num_pre_bev_layer = num_pre_bev_layer

    # @auto_fp16(apply_to=('query', 'key', 'value'))
    # @run_time('BEVTransformerEncoder_point_sampling')
    @force_fp32(apply_to=('reference_points', 'img_metas'))  # This function must use fp32 to perform matrix multi!!!
    def point_sampling(self, reference_points, pc_range, device, img_metas, gt_bboxes_3d=None, dataset_type='nuscenes'):

        file_names = img_metas[0]['filename']

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])

        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                         (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                         (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                         (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        if digit_version(TORCH_VERSION) == digit_version('1.8.1'):
            res = np.matmul(lidar2img.cpu().numpy(), reference_points.cpu().numpy())
            reference_points_cam = torch.from_numpy(res).squeeze(-1).to(device)
        else:
            reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                                reference_points.to(torch.float32)).squeeze(-1)

        eps = 1e-5
        mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        # print('reference_points_cam[..., 0:2]',reference_points_cam[..., 0:2])

        # TODO use ori_shape
        # print('reference_points_cam', reference_points_cam.shape)
        # print( img_metas[0]['ori_shape'])
        if len(list(set(img_metas[0]['ori_shape']))) > 1:  # if cams have different input shape
            for i, ori_shape in enumerate(img_metas[0]['ori_shape']):
                reference_points_cam[..., i, :, 0] /= ori_shape[1]
                reference_points_cam[..., i, :, 1] /= ori_shape[0]
        else:
            reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
            reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
        # reference_points_cam = (reference_points_cam - 0.5) * 2

        mask = (mask & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            mask = torch.nan_to_num(mask)
        else:
            mask = mask.new_tensor(np.nan_to_num(mask.cpu().numpy()))
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        mask = mask.permute(2, 1, 3, 0, 4).squeeze(-1)
        return reference_points_cam, mask

    @auto_fp16()
    # @run_time('BEVEncoder')
    def forward(self,
                query,
                key,
                value,
                *args,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                reg_branches=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = query
        intermediate = []

        reference_points_cam, mask = self.point_sampling(ref_3d, self.pc_range, query.device, kwargs['img_metas'])

        shift_ref_2d = ref_2d
        shift_ref_2d[..., :] += shift

        for lid, layer in enumerate(self.layers):

            if lid >= self.num_pre_bev_layer:
                prev_bev = None
                cur_ref_2d = ref_2d
            else:
                cur_ref_2d = shift_ref_2d
            output = layer(query,
                           key,
                           value,
                           *args,
                           bev_pos=bev_pos,
                           ref_2d=cur_ref_2d,
                           ref_3d=ref_3d,
                           bev_h=bev_h,
                           bev_w=bev_w,
                           spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index,
                           reference_points_cam=reference_points_cam,
                           mask=mask,
                           bev_queue=prev_bev,
                           **kwargs)

            query = output

            output = output.permute(1, 0, 2)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output