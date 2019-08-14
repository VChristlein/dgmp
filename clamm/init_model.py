import torch.nn as nn
from torchvision import models
import pooling
from poolnet import PoolNet

def set_grads(model, requires_grad=False):
    """ returns list of parameters that previously
        dont required any gradiends
    """
    old_non_grads = []
    old_non_grads_names = []
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            old_non_grads.append(param)
            old_non_grads_names.append(name)
        param.requires_grad = requires_grad
    return old_non_grads, old_non_grads_names

def initialize_model(model_name, n_classes,
                     use_pretrained=True,
                     pool_type='gmp',
                     gmp_lambda=1000,
                     lse_r=10,
                     last_conv_stride=2,
                     normalize='l2'
                    ):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model = None
    if model_name == "resnet18":
        model = models.resnet18(pretrained=use_pretrained)
    elif model_name == "resnet50":
        if last_conv_stride == 2:
            model = models.resnet50(pretrained=use_pretrained)
            if  pool_type != 'avg' and pool_type != 'max':
                print('WARNING: your network might not learn since you dont do'
                  ' l2-normalization (at least gmp wont learn) but maybe this !')
            
            model.avgpool = pooling.getPool(pool_type,
                                            gmp_lambda=gmp_lambda,
                                            lse_r=lse_r) 
        else:
            model = PoolNet(n_classes, pool_type, last_conv_stride=last_conv_stride,
                            pretrained=use_pretrained, normalize='none')
    elif model_name == 'poolnet':
        model = PoolNet(n_classes, pool_type, gmp_lambda=gmp_lambda,
                        lse_r=lse_r, last_conv_stride=last_conv_stride, 
                        pretrained=use_pretrained, normalize=normalize)
    else:
        raise ValueError('{} currently not supported'.format(model_name))

#    set_grads(model, requires_grad=False)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model

