import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class NormL2Loss(nn.Module):
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 weight=1.0,
                 **kwargs,
                 ):
        super(NormL2Loss, self).__init__()
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
        norms_S = torch.stack([preds_S.max(1).values.abs(), preds_S.min(1).values.abs()], dim=1).max(1).values
        norms_T = torch.stack([preds_T.max(1).values.abs(), preds_T.min(1).values.abs()], dim=1).max(1).values
        preds_S = preds_S /norms_S.unsqueeze(1)
        preds_T = preds_T /norms_T.unsqueeze(1)
        loss = ((preds_S - preds_T)**2).sum()  #.view(N,C,H*W).sum()  #.mean(-1).sum()

        return self.loss_weight * loss



