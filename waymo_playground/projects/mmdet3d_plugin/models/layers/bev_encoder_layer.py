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


import copy
import warnings
import torch
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER.register_module()
class BEVEncoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVEncoderLayer, self).__init__(attn_cfgs=attn_cfgs,
                                              feedforward_channels=feedforward_channels,
                                              ffn_dropout=ffn_dropout,
                                              operation_order=operation_order,
                                              act_cfg=act_cfg,
                                              norm_cfg=norm_cfg,
                                              ffn_num_fcs=ffn_num_fcs,
                                              **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(['self_attn', 'norm', 'cross_attn', 'ffn'])

    # @run_time('BEVEncoderLayer')
    # @auto_fp16(apply_to=('query', 'key', 'value', 'bev_pos', 'query_pos', 'key_padding_mask'))
    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                bev_queue=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'
        for layer in self.operation_order:
            if layer == 'self_attn':
                if bev_queue is not None:
                    temp_key = temp_value = bev_queue
                else:
                    temp_key = temp_value = query
                query = self.attentions[attn_index](query,
                                                    temp_key,
                                                    temp_value,
                                                    identity if self.pre_norm else None,
                                                    query_pos=bev_pos,
                                                    key_pos=bev_pos,
                                                    attn_mask=attn_masks[attn_index],
                                                    key_padding_mask=query_key_padding_mask,
                                                    reference_points=ref_2d,
                                                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                                                    level_start_index=torch.tensor([0], device=query.device),
                                                    **kwargs)

                if bev_queue is not None and query.size(1) > 1:
                    assert False
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](query,
                                                    key,
                                                    value,
                                                    identity if self.pre_norm else None,
                                                    query_pos=query_pos,
                                                    key_pos=key_pos,
                                                    reference_points=ref_3d,
                                                    reference_points_cam=reference_points_cam,
                                                    mask=mask,
                                                    attn_mask=attn_masks[attn_index],
                                                    key_padding_mask=key_padding_mask,
                                                    spatial_shapes=spatial_shapes,
                                                    level_start_index=level_start_index,
                                                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query