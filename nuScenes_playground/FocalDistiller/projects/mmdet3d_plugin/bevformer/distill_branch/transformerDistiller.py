# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
# from .temporal_self_attention import TemporalSelfAttention
# from .spatial_cross_attention import MSDeformableAttention3D
from .decoderDistiller import CustomMSDeformableAttentionDistiller
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16


@TRANSFORMER.register_module()
class PerceptionTransformerDistiller(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 # encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformerDistiller, self).__init__(**kwargs)
        # self.encoder = build_transformer_layer_sequence(encoder) # encoder: 'BEVFormerEncoder'
        self.decoder = build_transformer_layer_sequence(decoder) # decoder: 'DetectionTransformerDecoder'
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        # self.level_embeds = nn.Parameter(torch.Tensor(
        #     self.num_feature_levels, self.embed_dims))
        # self.cams_embeds = nn.Parameter(
        #     torch.Tensor(self.num_cams, self.embed_dims))

        self.reference_points = nn.Linear(self.embed_dims, 3)

        # self.can_bus_mlp = nn.Sequential(
        #     nn.Linear(18, self.embed_dims // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.embed_dims // 2, self.embed_dims),
        #     nn.ReLU(inplace=True),
        # )
        # if self.can_bus_norm:
        #     self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            # if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
            #         or isinstance(m, CustomMSDeformableAttention):
            if isinstance(m, CustomMSDeformableAttentionDistiller):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        #normal_(self.level_embeds)
        #normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        #xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)


    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                bev_embed_T,
                bev_embed_S,
                object_query_embed,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        # bs, embed_dims, bev_h, bev_w -> bs, bev_h*bev_w, embed_dims
        bs, embed_dims, bev_h_T, bev_w_T = bev_embed_T.shape
        bev_embed_T = bev_embed_T.view(bs, embed_dims, bev_h_T * bev_w_T).permute(2, 0, 1)
        _, __, bev_h_S, bev_w_S = bev_embed_S.shape
        bev_embed_S = bev_embed_S.view(bs, embed_dims, bev_h_S * bev_w_S).permute(2, 0, 1)
        query_pos, query = torch.split( 
            object_query_embed, self.embed_dims, dim=1) # torch.Size([900, 512]) -> 2 * torch.Size([900, 256])
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1) 
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)

        inter_states, inter_references, sampling_feats_pairs = self.decoder( # DetectionTransformerDecoder
            query=query, # torch.Size([900, 1, 256])
            key=None,
            value=bev_embed_T, # torch.Size([bev_h*bev_w, 1, 256])
            value_T=bev_embed_T,
            value_S=bev_embed_S, 
            query_pos=query_pos, # torch.Size([900, 1, 256])
            reference_points=reference_points, # torch.Size([1, 900, 3])
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h_T, bev_w_T]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return inter_states, init_reference_out, inter_references_out, sampling_feats_pairs
