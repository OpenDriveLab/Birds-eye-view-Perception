import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES
from torch.cuda.amp.autocast_mode import autocast


@DISTILL_LOSSES.register_module()
class BCELoss(nn.Module):

    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation
     <https://arxiv.org/abs/2011.13256>`_.
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name(str): 
        tau (float, optional): Temperature coefficient. Defaults to 1.0.
        weight (float, optional): Weight of loss.Defaults to 1.0.
        
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 tau=1.0,
                 weight=1.0,
                 ):
        super(BCELoss, self).__init__()
        self.tau = tau
        self.loss_weight = weight
    
        if student_channels != teacher_channels:
            self.align = nn.Sequential(
                nn.BatchNorm2d(num_features=student_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.align = None


    def forward(self,
                preds_S,
                preds_T):
        """Forward function."""
        assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'
        N,C,W,H = preds_S.shape

        if self.align is not None:
            preds_S = self.align(preds_S)

        return self.get_depth_loss(preds_S, preds_T)
    
    
    def get_depth_loss(self, depth_preds, depth_labels):
        depth_labels = self.get_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, 118)

        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, depth_labels.shape[0])
        return self.loss_weight * depth_loss

    def get_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, H, W = gt_depths.shape

        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=119).view(-1, 119)[:, 1:]
        return gt_depths.float()