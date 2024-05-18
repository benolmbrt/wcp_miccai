import torch
from torch import nn
import numpy as np
from typing import Tuple

from ..nets.segmentation.wrappers.get_backbone import get_backbone
from monai.networks.blocks import ADN

"""
Multi-head segmentation wrapper allowing to compute predictive intervals for volumes. 
"""

class MultiHead_MonaiWrapper(nn.Module):
    def __init__(self,
                 backbone: str,
                 dim: int,
                 in_channels: int,
                 n_classes: int,
                 dropout: float = 0.,
                 norm: str = 'batch',
                 output_features: int = 32,
                 image_size: Tuple[int, ...] = None):
        super().__init__()

        self.backbone = backbone
        self.dim = dim
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.norm = norm
        self.dropout = dropout
        self.image_size = image_size
        self.output_features = output_features

        # define basic operators
        assert self.dim in [2, 3], 'Supported dimensions are [2, 3] but you provided {}'.format(self.dim)
        # get monai backbone
        self.backbone = get_backbone(backbone=self.backbone,
                                     dim=self.dim,
                                     in_channels=self.in_channels,
                                     n_classes=self.output_features,
                                     dropout=self.dropout,
                                     normalization=self.norm,
                                     image_size=self.image_size)

        self.norm_operator = ADN(in_channels=self.output_features, norm=self.norm,
                                 norm_dim=self.dim, dropout=self.dropout, dropout_dim=self.dim)

        operator = nn.Conv2d if dim == 2 else nn.Conv3d

        # output heads
        self.head_lower = operator(kernel_size=1, stride=1, in_channels=self.output_features, out_channels=self.n_classes)
        self.head_mean = operator(kernel_size=1, stride=1, in_channels=self.output_features, out_channels=self.n_classes)
        self.head_upper = operator(kernel_size=1, stride=1, in_channels=self.output_features, out_channels=self.n_classes)

    def forward(self, input: torch.Tensor):
        feat = self.norm_operator(self.backbone(input))
        mean = self.head_mean(feat)
        lower = self.head_lower(feat)
        upper = self.head_upper(feat)

        out_dict = {'logits': mean,
                    'upper': upper,
                    'lower': lower}
        #
        return out_dict

    def prediction(self,
                   x: torch.Tensor,
                   **kwargs) -> np.ndarray:

        pred_dict = self(x)
        logits = pred_dict['logits']

        avg_prob = torch.sigmoid(logits)
        upper_prob = torch.sigmoid(pred_dict['upper'])
        lower_prob = torch.sigmoid(pred_dict['lower'])

        foreground_classes = range(1, self.n_classes)
        out_dict = {'logits': pred_dict['logits']}

        for n in foreground_classes:
            mean_mask = (avg_prob[:, n] >= 0.5).long().unsqueeze(1)
            upper_mask = (upper_prob[:, n] >= 0.5).long().unsqueeze(1)
            lower_mask = (lower_prob[:, n] >= 0.5).long().unsqueeze(1)

            bounds = mean_mask + upper_mask + lower_mask
            out_dict[f'uncertainty_bounds_{n}'] = bounds
            out_dict[f'uncertainty_lower_{n}'] = lower_prob[:, n].unsqueeze(1)
            out_dict[f'uncertainty_upper_{n}'] = upper_prob[:, n].unsqueeze(1)
            out_dict[f'uncertainty_mean_{n}'] = avg_prob[:, n].unsqueeze(1)

        return out_dict
