# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.utils import digit_version
from torch import Tensor

from mmdet.registry import MODELS

MODELS.register_module('Linear', module=nn.Linear)


@MODELS.register_module(name='NormedLinear')
class NormedLinear(nn.Linear):
    """Normalized Linear Layer.

    Args:
        tempeature (float, optional): Tempeature term. Defaults to 20.
        power (int, optional): Power term. Defaults to 1.0.
        eps (float, optional): The minimal value of divisor to
             keep numerical stability. Defaults to 1e-6.
    """

    def __init__(self,
                 *args,
                 tempearture: float = 20,
                 power: int = 1.0,
                 eps: float = 1e-6,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tempearture = tempearture
        self.power = power
        self.eps = eps
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights."""
        nn.init.normal_(self.weight, mean=0, std=0.01)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for `NormedLinear`."""
        weight_ = self.weight / (
            self.weight.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x / (x.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x_ * self.tempearture

        return F.linear(x_, weight_, self.bias)


@MODELS.register_module(name='NormedConv2d')
class NormedConv2d(nn.Conv2d):
    """Normalized Conv2d Layer.

    Args:
        tempeature (float, optional): Tempeature term. Defaults to 20.
        power (int, optional): Power term. Defaults to 1.0.
        eps (float, optional): The minimal value of divisor to
             keep numerical stability. Defaults to 1e-6.
        norm_over_kernel (bool, optional): Normalize over kernel.
             Defaults to False.
    """

    def __init__(self,
                 *args,
                 tempearture: float = 20,
                 power: int = 1.0,
                 eps: float = 1e-6,
                 norm_over_kernel: bool = False,
                 learnable_temp=False,
                 init_bias=0.0,
                 learnable_init_value=4.5,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tempearture = tempearture
        self.power = power
        self.norm_over_kernel = norm_over_kernel
        self.eps = eps
        self.learnable_temp = learnable_temp
        self.init_bias = init_bias
        self.init_lr_temp=learnable_init_value
        if self.learnable_temp is True:
            self.tempearture = torch.nn.Parameter(torch.tensor(1.0,device='cuda'))

    def init_weights(self):
        nn.init.normal_(self.weight, mean=0, std=0.01)
        if self.learnable_temp is True:
            self.tempearture.data.fill_(self.init_lr_temp)
        if self.bias is not None:
            nn.init.constant_(self.bias, self.init_bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for `NormedConv2d`."""
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
            if digit_version(torch.__version__) >= digit_version('1.8'):
                x_ = self._conv_forward(x_, weight_, self.bias)
            else:
                x_ = self._conv_forward(x_, weight_)
        return x_


@MODELS.register_module(name='WCLinear')
class WCLinear(nn.Linear):
    """Wrapped Cauchy Linear Layer.
     unofficial implementation of Boran Han 
     'Wrapped Cauchy Distributed Angular Softmax for Long-Tailed Visual Recognition' in ICML2023

    Args:
        tempeature (float, optional): Tempeature term. Default to 20.
        power (int, optional): Power term. Default to 1.0.
        eps (float, optional): The minimal value of divisor to
             keep numerical stability. Default to 1e-6.
    """

    def __init__(self, *args, power=1.0, eps=1e-6,init_bias=0.0,learnable_init_value=50, gamma = 1.0, s_trainable=True, **kwargs):
        super(WCLinear, self).__init__(*args, **kwargs)
        self.init_bias = init_bias
        self.power = power
        self.eps = eps
        self.s_trainable = s_trainable
        out_features = self.weight.shape[0]
        self.g = torch.nn.Parameter(gamma*torch.ones(out_features,device='cuda'), requires_grad=True) 
        self.init_lr_temp=learnable_init_value
        if self.s_trainable is True:
            self.s_ = torch.nn.Parameter(torch.tensor(0.0,device='cuda'),requires_grad=True)
        else:
            self.s_ = 300
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.weight, mean=0, std=0.01)
        if self.bias is not None:
            nn.init.constant_(self.bias, self.init_bias)
        if self.s_trainable is True:
            self.s_.data.fill_(self.init_lr_temp)

    def forward(self, x):
        weight_ = self.weight / (
            self.weight.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x / (x.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        cosine = F.linear(x_, weight_, self.bias)
        gamma =  1 / (1 + torch.exp (-self.g))
        logit =  1/2/3.14*(1-gamma**2)/(1+gamma**2-2*gamma*cosine)
        logit = (self.s_) * logit

        return logit