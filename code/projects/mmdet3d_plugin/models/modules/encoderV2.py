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
class BEVTransformerEncoderV2(TransformerLayerSequence):
    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 pc_range=None,
                 num_pre_bev_layer=None,
                 use_key_padding_mask=False,
                 return_intermediate=False,
                 dataset_type='nuscenes',
                 **kwargs):

        super(BEVTransformerEncoderV2, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.use_key_padding_mask = use_key_padding_mask
        self.dataset_type = dataset_type
        self.count = 0

    @force_fp32(apply_to=('reference_points', 'img_metas'))  # This function must use fp32!!!
    def point_sampling(self,
                       reference_points,
                       pc_range,
                       device,
                       img_metas,
                       gt_bboxes_3d=None,
                       dataset_type='nuscenes',
                       bev_h=200,
                       bev_w=200):

        file_names = img_metas[0]['filename']
        # print(file_names)
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        # if dataset_type == 'nuscenes':
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                         (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                         (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                         (pc_range[5] - pc_range[2]) + pc_range[2]
        # elif dataset_type == 'waymo':
        #     # [x, y] = [y, -x]
        #     # use for zero shot
        #     x = (reference_points[..., 1:2] *
        #                                    (pc_range[3] - pc_range[0]) + pc_range[0])
        #     y = -(reference_points[..., 0:1] *
        #                                  (pc_range[4] - pc_range[1]) + pc_range[1])
        #     reference_points[..., 2:3] = reference_points[..., 2:3] * \
        #                                  (pc_range[5] - pc_range[2]) + pc_range[2]
        #     reference_points[..., 0:1] = x
        #     reference_points[..., 1:2] = y

        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32), reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # TODO use ori_shape
        # print('reference_points_cam', reference_points_cam.shape)

        if len(list(set(img_metas[0]['ori_shape']))) > 1:  # if cams have different input shape
            for i, ori_shape in enumerate(img_metas[0]['ori_shape']):
                reference_points_cam[..., i, :, 0] /= ori_shape[1]
                reference_points_cam[..., i, :, 1] /= ori_shape[0]
        else:
            reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
            reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

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
        self.key_padding_mask = mask.permute(1, 0, 3, 2).reshape(B, -1, bev_h * bev_w)  #.cpu().numpy()
        self.key_padding_mask = self.key_padding_mask.permute(0, 2, 1).sum(-1)
        self.key_padding_mask = (self.key_padding_mask == 0)
        if not self.use_key_padding_mask:
            self.key_padding_mask = None

        # save_tensor(tmp, 'tmp3_waymo2.png')
        # imgs = []
        # for each in file_names:
        #     img = mmcv.imread(each)
        #     imgs.append(img)
        # for center in gt_bboxes_3d[0].gravity_center:
        #
        #     points = [center[0], center[1], center[2]]
        #
        #     points.append(1)
        #
        #     points = torch.tensor(points, device=lidar2img.device)
        #     points = points.view(1, 1, 1, 1, 4).repeat(
        #         1, 1, num_cam, num_query, 1).unsqueeze(-1)
        #     points = torch.matmul(lidar2img[0:1], points).squeeze(-1)
        #
        #     eps = 1e-5
        #
        #     points = points[..., 0:2] / \
        #              torch.maximum(points[..., 2:3],
        #                            torch.ones_like(points[..., 2:3]) * eps)
        #     points = points.view(1, num_cam, num_query, 2)
        #     for k in range(num_cam):
        #         per_point = points[0, k, 0, :]
        #         per_point = per_point.cpu().numpy()
        #         try:
        #             per_point = (int(per_point[0]), int(per_point[1]))
        #         except:
        #             continue
        #
        #         if 0 < per_point[0] < img_metas[0]['img_shape'][0][1] and 0 < per_point[1] < \
        #                 img_metas[0]['img_shape'][0][0]:
        #             font = cv.FONT_HERSHEY_SIMPLEX
        #             cv.circle(imgs[k], per_point, radius=5,
        #                       color=(255, 0, 0), thickness=4)
        #             text = '%.0f,%.0f' % (center[0].item(), center[1].item())
        #             print(text)
        #             cv.putText(imgs[k], text, per_point, font, 1.2, (255, 255, 255), 2)
        # first_row = np.hstack([imgs[1], imgs[0], imgs[2]])
        # sencond_row = np.hstack([imgs[3], imgs[4], imgs[5]])
        # view = np.vstack([first_row, sencond_row])
        # mmcv.imwrite(view, f'{time.time()}.png')
        # exit()
        return reference_points_cam, mask

    @auto_fp16()
    #@run_time('BEVEncoderV2')
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
                gt_bboxes_3d=None,
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

        reference_points_cam, mask = self.point_sampling(ref_3d,
                                                         self.pc_range,
                                                         query.device,
                                                         kwargs['img_metas'],
                                                         gt_bboxes_3d=gt_bboxes_3d,
                                                         dataset_type=self.dataset_type,
                                                         bev_h=bev_h,
                                                         bev_w=bev_w)

        shift_ref_2d = ref_2d
        shift_ref_2d[..., :] += shift

        if prev_bev is not None:
            # using current bev and previous bev to build bev queue.
            bev_queue = torch.cat([prev_bev, query], 1)
            cur_ref_2d = torch.cat([shift_ref_2d, ref_2d], 0)
        else:
            bev_queue = None
            cur_ref_2d = ref_2d

        for lid, layer in enumerate(self.layers):
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
                           key_padding_mask=self.key_padding_mask,
                           mask=mask,
                           bev_queue=bev_queue,
                           **kwargs)

            query = output

            output = output.permute(1, 0, 2)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
