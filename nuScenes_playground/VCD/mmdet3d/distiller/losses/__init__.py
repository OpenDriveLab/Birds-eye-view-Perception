from .cwd import ChannelWiseDivergence
from .norml2loss import NormL2Loss
from .l1loss import L1Loss
from .l2loss import L2Loss 
from .klloss import KLLoss
from .bceloss import BCELoss
from .maskedbceloss import MaskedBCELoss
__all__ = [
    'ChannelWiseDivergence',
    'NormL2Loss',
    'L1Loss',
    'L2Loss',
    'KLLoss',
    'BCELoss',
    'MaskedBCELoss',
]
