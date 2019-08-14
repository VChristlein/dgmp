import torch
import torch.nn as nn

import torchvision.models as models
import config
from pooling import GMP, WeightedAvgPool2d, GeM, LSEPool, GMPFixedParameter, MixedPool
from resnet import resnet50


class ResNet50Encoder(nn.Module):
    """
    Encoding network based on ResNet-50.

    Args:
         pooling (str): denotes the pooling technique.

    """
    def __init__(self, pool='avg', loss={'triplet'}, gmp_lambda=1e6,
                 conv_dim=2048, last_filter_size=(8, 8), lse_r=10, 
                 last_conv_stride=2, normalize='l2'):
        super().__init__()
        self.pooling_type = pool
        self.loss = loss
        self.feature_extractor = resnet50(pretrained=True, last_conv_stride=last_conv_stride)
        if pool == 'gmp':
            self.pool = GMP(lamb=gmp_lambda)
        elif pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'weighted_avg':
            self.pool = WeightedAvgPool2d(init_weight=1, dim=self.dim_encoding)
        elif pool == 'gem':
            self.pool = GeM()
        elif pool == 'conv':
            self.pool = nn.Conv2d(2048, conv_dim, kernel_size=last_filter_size)
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pool == 'lse':
            self.pool = LSEPool(lse_r)
        elif pool == 'gmp-fix':
            self.pool = GMPFixedParameter(lamb=gmp_lambda)
        elif pool == 'mixed-pool':
            self.pool = MixedPool(0.5)
        else:
            raise RuntimeError('{} is not a valid pooling strategy.'.format(pool))
        
        if normalize == 'l2':
            self.normalize = nn.functional.normalize
        else:
            self.normalize = lambda x : x

        print(self)

    def forward(self, x):
        f = self.feature_extractor(x)
        y = self.pool(f)
        y = self.normalize(y)
        y = y.view(y.size(0), -1)
        if self.loss == {'triplet', 'pool'}:
            return y, f
        else: 
            return y
#        elif self.loss == {'triplet'}:
#            return y

    def extract_features(self, x):
        f = self.feature_extractor(x)
        y = self.pool(f)
        y = self.normalize(y)
        y = y.view(y.size(0), -1)
        return y


# class AlexnetEncoder(nn.Module):
#
#     def __init__(self, feature_extractor='resnet50', gmp_lambda=1e6, pool='gmp', use_fc=False):
#         super().__init__()
#         self.pooling_type = pool
#         self.extractor_type = feature_extractor
#
#         self.feature_extractor = models.alexnet(pretrained=True).features
#
#         self.feature_dim = 256
#
#         if pool == 'gmp':
#             # dim_encoding = 256
#             # self.fc_dim = dim_encoding
#             self.head = nn.Sequential(
#                 # nn.Conv2d(dim_features, dim_encoding, 1),
#                 # nn.BatchNorm2d(dim_encoding),
#                 # nn.ReLU(inplace=True),
#                 GMP(lamb=gmp_lambda),
#             )
#         elif pool == 'avg':
#             self.head = nn.AdaptiveAvgPool2d(1)
#         elif pool == 'max':
#             self.head = nn.AdaptiveMaxPool2d(1)
#         if use_fc:
#             fc = nn.Sequential(
#                 encoding.nn.View(-1, self.feature_dim),
#                 nn.Linear(self.feature_dim, 1024),
#                 nn.BatchNorm1d(1024),
#                 nn.ReLU(),
#                 nn.Linear(1024, 128)
#             )
#             self.head.add_module('fc', fc)
#
#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = self.pool(x)
#         return x
#
#     def extract_features(self, x):
#         return self.forward(x)
