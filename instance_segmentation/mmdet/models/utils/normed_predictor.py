# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import CONV_LAYERS
from torch.nn import Parameter

from .builder import LINEAR_LAYERS


@LINEAR_LAYERS.register_module(name='NormedLinear')
class NormedLinear(nn.Linear):
    """Normalized Linear Layer.

    Args:
        tempeature (float, optional): Tempeature term. Default to 20.
        power (int, optional): Power term. Default to 1.0.
        eps (float, optional): The minimal value of divisor to
             keep numerical stability. Default to 1e-6.
    """

    def __init__(self, *args, tempearture=20, power=1.0, eps=1e-6, init_bias=0,learnable_temp=False,learnable_init_value=4.0,**kwargs):
        super(NormedLinear, self).__init__(*args, **kwargs)
        self.learnable_temp =learnable_temp
        if self.learnable_temp is True:
            self.tempearture = Parameter(torch.FloatTensor(1).cuda())
            self.learnable_init_value = learnable_init_value
        else:
            self.tempearture = tempearture
        self.power = power
        self.eps = eps
        self.init_bias = init_bias
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.weight, mean=0, std=0.01)
        if self.bias is not None:
            nn.init.constant_(self.bias, self.init_bias)
        if self.learnable_temp is True:
            self.tempearture.data.fill_(self.learnable_init_value)

    def forward(self, x):
        weight_ = self.weight / (
            self.weight.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x / (x.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
#         x_ = x_ * self.tempearture
        
        if self.learnable_temp is True:
            x_ = x_ * (self.tempearture**2)
        else:
            x_ = x_ * self.tempearture

        return F.linear(x_, weight_, self.bias)


@CONV_LAYERS.register_module(name='NormedConv2d')
class NormedConv2d(nn.Conv2d):
    """Normalized Conv2d Layer.

    Args:
        tempeature (float, optional): Tempeature term. Default to 20.
        power (int, optional): Power term. Default to 1.0.
        eps (float, optional): The minimal value of divisor to
             keep numerical stability. Default to 1e-6.
        norm_over_kernel (bool, optional): Normalize over kernel.
             Default to False.
    """

    def __init__(self,
                 *args,
                 tempearture=20,
                 power=1.0,
                 eps=1e-6,
                 norm_over_kernel=False,
                 learnable_temp=False,
                 learnable_init_value=4.5,
                 init_bias=0,
                 **kwargs):
        super(NormedConv2d, self).__init__(*args, **kwargs)
        self.learnable_temp =learnable_temp
        self.init_bias = init_bias
        if self.learnable_temp is True:
            self.tempearture = Parameter(torch.FloatTensor(1).cuda())
            self.learnable_init_value = learnable_init_value
        else:
            self.tempearture = tempearture
        self.power = power
        self.norm_over_kernel = norm_over_kernel
        self.eps = eps
        
    def init_weights(self):
        nn.init.normal_(self.weight, mean=0, std=0.01)
        if self.bias is not None:
            nn.init.constant_(self.bias, self.init_bias)
        if self.learnable_temp is True:
            self.tempearture.data.fill_(self.learnable_init_value)

    def forward(self, x):
        if not self.norm_over_kernel:
            weight_ = self.weight / (
                self.weight.norm(dim=1, keepdim=True).pow(self.power) +
                self.eps)
        else:
            weight_ = self.weight / (
                self.weight.view(self.weight.size(0), -1).norm(
                    dim=1, keepdim=True).pow(self.power)[..., None, None] +
                self.eps)
        x_ = x / (x.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        if self.learnable_temp is True:
            x_ = x_ * (self.tempearture**2)
        else:
            x_ = x_ * self.tempearture
        
        

        if hasattr(self, 'conv2d_forward'):
            x_ = self.conv2d_forward(x_, weight_)
        else:
            if torch.__version__ >= '1.8':
                x_ = self._conv_forward(x_, weight_, self.bias)
            else:
                x_ = self._conv_forward(x_, weight_)
        return x_
