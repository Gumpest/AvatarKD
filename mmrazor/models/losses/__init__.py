# Copyright (c) OpenMMLab. All rights reserved.
from .cwd import ChannelWiseDivergence
from .kl_divergence import KLDivergence
from .relational_kd import AngleWiseRKD, DistanceWiseRKD
from .weighted_soft_label_distillation import WSLD
from .akd_cwd import AKD_ChannelWiseDivergence
from .akd_maskd import AKD_MasKDLoss
from .akd_mgd import AKD_MGDLoss


__all__ = [
    'ChannelWiseDivergence', 'KLDivergence', 'AngleWiseRKD', 'DistanceWiseRKD',
    'WSLD', 'AKD_ChannelWiseDivergence', 'AKD_MasKDLoss', 'AKD_MGDLoss'
]
