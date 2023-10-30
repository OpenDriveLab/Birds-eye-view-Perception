import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class L1Loss(nn.Module):
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 weight=1.0,
                 **kwargs,
                 ):
        super(L1Loss, self).__init__()
        self.loss_weight = weight
    
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None


    def forward(self,
                preds_S,
                preds_T):
        """Forward function."""
        assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'
        N, C, H, W = preds_S.shape

        if self.align is not None:
            preds_S = self.align(preds_S)

        loss = ((preds_S - preds_T).abs()).sum()  #.view(N,C,H*W).sum()  #.mean(-1).sum()
        return self.loss_weight * loss



