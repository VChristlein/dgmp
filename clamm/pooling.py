import torch
import torch.nn as nn
import torch.nn.functional as F

def getPool(pool_type='avg', gmp_lambda=1e3, lse_r=10):
    """
    params
        pool_type: the allowed pool types
        gmp_lambda: the initial regularization parameter for GMP
        lse_r: the initial regularization parameter for LSE
    """
    if pool_type == 'gmp':
        pool_layer = GMP(lamb=gmp_lambda)
    elif pool_type == 'avg':
        pool_layer = nn.AdaptiveAvgPool2d(1)
    elif pool_type == 'max':
        pool_layer = nn.AdaptiveMaxPool2d(1)
    elif pool_type == 'mixed-pool':
        pool_layer = MixedPool(0.5)
    elif pool_type == 'lse':
        pool_layer = LSEPool(lse_r)
    else:
        raise RuntimeError('{} is not a valid pooling'
                           ' strategy.'.format(pool_type))

    return pool_layer

class GMP(nn.Module):
    """ Generalized Max Pooling
    """
    def __init__(self, lamb):
        super().__init__()
        self.lamb = nn.Parameter(lamb * torch.ones(1))
        #self.inv_lamb = nn.Parameter((1./lamb) * torch.ones(1))

    def forward(self, x):
        B, D, H, W = x.shape
        N = H * W
        identity = torch.eye(N).cuda()
        # reshape x, s.t. we can use the gmp formulation as a global pooling operation
        x = x.view(B, D, N)
        x = x.permute(0, 2, 1)
        # compute the linear kernel
        K = torch.bmm(x, x.permute(0, 2, 1))
        # solve the linear system (K + lambda * I) * alpha = ones
        A = K + self.lamb * identity
        o = torch.ones(B, N, 1).cuda()
        #alphas, _ = torch.gesv(o, A) # tested using pytorch 1.0.1
        alphas, _ = torch.solve(o, A) # tested using pytorch 1.2.0
        alphas = alphas.view(B, 1, -1)        
        xi = torch.bmm(alphas, x)
        xi = xi.view(B, -1)
        return xi

class MixedPool(nn.Module):
    def __init__(self, a):
        super(MixedPool, self).__init__()
        self.a = nn.Parameter(a * torch.ones(1))

    def forward(self, x):
        return self.a * F.adaptive_max_pool2d(x, 1) + (1 - self.a) * F.adaptive_avg_pool2d(x, 1)
    
class LSEPool(nn.Module):
    """
    Learnable LSE pooling with a shared parameter
    """

    def __init__(self, r):
        super(LSEPool, self).__init__()
        self.r = nn.Parameter(torch.ones(1) * r)

    def forward(self, x):
        s = (x.size(2) * x.size(3))
        x_max = F.adaptive_max_pool2d(x, 1)
        exp = torch.exp(self.r * (x - x_max))
        sumexp = 1 / s * torch.sum(exp, dim=(2, 3))
        sumexp = sumexp.view(sumexp.size(0), -1, 1, 1)
        logsumexp = x_max + 1 / self.r * torch.log(sumexp)
        return logsumexp

