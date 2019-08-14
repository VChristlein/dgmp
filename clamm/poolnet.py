import torch.nn as nn
from torchvision import models
import pooling

from pooling import getPool
from resnet import resnet50

class PoolNet(nn.Module):
    def __init__(self, n_classes=12, pool_type='avg', gmp_lambda=1e6,
                 lse_r=10, last_conv_stride=2, 
                 pretrained=True, input_channels=3, normalize='l2'):
        super().__init__()
        self.pooling_type = pool_type
        # TODO: make it more general
        # -> difficult since pooling layers are named differently in the
        # architectures
        self.feature_extractor = resnet50(pretrained=pretrained, 
                                          last_conv_stride=last_conv_stride, 
                                          input_channels=input_channels)
        self.pool = getPool(pool_type, gmp_lambda, lse_r)
        # Note: this currently only applies to ResNet50
        self.fc = nn.Linear(2048, n_classes)
        
        if normalize == 'l2':
            self.normalize = nn.functional.normalize
        else:
            self.normalize = lambda x : x

    def forward(self, x):
        f = self.feature_extractor(x)
        y = self.pool(f)
        y = self.normalize(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y

