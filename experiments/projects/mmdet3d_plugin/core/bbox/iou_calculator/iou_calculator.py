# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core.bbox import bbox_overlaps
from mmdet.core.bbox.iou_calculators.builder import IOU_CALCULATORS
from mmdet3d.core.bbox.structures import get_box_type


def bbox_overlaps_nearest_3d_with_let(bboxes1,
                             bboxes2,
                             mode='iou',
                             is_aligned=False,
                             coordinate='lidar'):
    """Calculate nearest 3D IoU with LET.
       bboxes1 is assumes to be GT

    Note:
        This function first finds the nearest 2D boxes in bird eye view
        (BEV), and then calculates the 2D IoU using :meth:`bbox_overlaps`.
        Ths IoU calculator :class:`BboxOverlapsNearest3D` uses this
        function to calculate IoUs of boxes.

        If ``is_aligned`` is ``False``, then it calculates the ious between
        each bbox of bboxes1 and bboxes2, otherwise the ious between each
        aligned pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (torch.Tensor): shape (N, 7+C) [x, y, z, h, w, l, ry, v].
        bboxes2 (torch.Tensor): shape (M, 7+C) [x, y, z, h, w, l, ry, v].
        mode (str): "iou" (intersection over union) or iof
            (intersection over foreground).
        is_aligned (bool): Whether the calculation is aligned

    Return:
        torch.Tensor: If ``is_aligned`` is ``True``, return ious between \
            bboxes1 and bboxes2 with shape (M, N). If ``is_aligned`` is \
            ``False``, return shape is M.
    """
    assert bboxes1.size(-1) == bboxes2.size(-1) >= 7

    # LET transform
    N = bboxes1.size(0)
    M = bboxes2.size(0)
    pred_center = bboxes2[:, :3]
    gt_center = bboxes1[:, :3]
    pred_range_sqr = pred_center.pow(2).sum(-1) + 1e-6
    gt_range = gt_center.pow(2).sum(-1).sqrt() + 1e-6
    range_multiplier = (pred_center.view(1, M, -1) * gt_center.view(N, 1, -1)).sum(-1) / pred_range_sqr.view(1, M)
    bboxes2_let = bboxes2.clone().view(1, M, -1).repeat(N, 1, 1)
    bboxes2_let[:, :, :3] = bboxes2_let[:, :, :3] * range_multiplier.view(N, M, 1)
    #iou_multiplier = (range_multiplier - 1).abs() * 10
    iou_multiplier = 1 - ((bboxes2_let[:, :, :3] - bboxes2.view(1, M, -1)[:, :, :3]).pow(2).sum(-1).sqrt() / (0.1 * gt_range.view(N, 1))).clamp(0, 1)
    iou_multiplier = iou_multiplier.view(1, -1)



    box_type, _ = get_box_type(coordinate)
    bboxes1 = box_type(bboxes1, box_dim=bboxes1.shape[-1])
    bboxes2 = box_type(bboxes2_let.view(M * N, -1), box_dim=bboxes2_let.shape[-1])
    bboxes1_bev = bboxes1.nearest_bev
    bboxes2_bev = bboxes2.nearest_bev
    ret = []
    for i in range(N):

        # Change the bboxes to bev
        # box conversion and iou calculation in torch version on CUDA
        # is 10x faster than that in numpy version

        ret_i = bbox_overlaps(
            bboxes1_bev[i:i+1, :], bboxes2_bev[i*M:(i+1)*M, :], mode=mode, is_aligned=is_aligned)
        ret_i = ret_i * iou_multiplier[:, i*M:(i+1)*M]
        ret.append(ret_i)
    ret = torch.cat(ret, dim=0)
    return ret

@IOU_CALCULATORS.register_module()
class BboxOverlapsNearest3D_with_let(object):
    """Nearest 3D IoU Calculator.
    Note:
        This IoU calculator first finds the nearest 2D boxes in bird eye view
        (BEV), and then calculates the 2D IoU using :meth:`bbox_overlaps`.
    Args:
        coordinate (str): 'camera', 'lidar', or 'depth' coordinate system.
    """

    def __init__(self, coordinate='lidar'):
        assert coordinate in ['camera', 'lidar', 'depth']
        self.coordinate = coordinate

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate nearest 3D IoU.
        Note:
            If ``is_aligned`` is ``False``, then it calculates the ious between
            each bbox of bboxes1 and bboxes2, otherwise it calculates the ious
            between each aligned pair of bboxes1 and bboxes2.
        Args:
            bboxes1 (torch.Tensor): shape (N, 7+N) [x, y, z, h, w, l, ry, v].
            bboxes2 (torch.Tensor): shape (M, 7+N) [x, y, z, h, w, l, ry, v].
            mode (str): "iou" (intersection over union) or iof
                (intersection over foreground).
            is_aligned (bool): Whether the calculation is aligned.
        Return:
            torch.Tensor: If ``is_aligned`` is ``True``, return ious between \
                bboxes1 and bboxes2 with shape (M, N). If ``is_aligned`` is \
                ``False``, return shape is M.
        """
        return bbox_overlaps_nearest_3d_with_let(bboxes1, bboxes2, mode, is_aligned,
                                        self.coordinate)

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(coordinate={self.coordinate}'
        return repr_str