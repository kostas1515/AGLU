'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
import math
try:
    import cbam 
except ImportError:
    from classification import cbam

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

class Unified(nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1,device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        lambda_param = torch.nn.init.uniform_(torch.empty(num_parameters, **factory_kwargs))
        kappa_param = torch.nn.init.uniform_(torch.empty(num_parameters, **factory_kwargs),a=-1.0,b=0.0) #works well with this init
        self.softplus = nn.Softplus(beta=-1.0)
        
        if num_parameters>1:
            self.lambda_param = nn.Parameter(lambda_param.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
            self.kappa_param = nn.Parameter(kappa_param.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        else:
            self.lambda_param = nn.Parameter(lambda_param)
            self.kappa_param = nn.Parameter(kappa_param)

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        l = torch.clamp(self.lambda_param,min=0.0001)
        p = torch.exp((1/l) * self.softplus((self.kappa_param*input) - torch.log(l)))
        
        return p

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)
    


class UniRect_static(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, lambda_param, kappa_param):
        # normal forward pass
#         input, lambda_param, kappa_param = input.detach(), lambda_param.detach(), kappa_param.detach()
        p  = torch.exp((1/lambda_param) * torch.nn.functional.softplus(kappa_param*input -torch.log(lambda_param), beta=-1.0, threshold=20))
        ctx.save_for_backward(input, lambda_param, kappa_param,p)
        out = p * input

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, lambda_param, kappa_param, p = ctx.saved_tensors
        sigmoidal_coeff = p/(lambda_param + torch.exp(kappa_param*input))

        part_grad_kappa = (input**2)*sigmoidal_coeff
        part_grad_lambda = (-input/lambda_param)*sigmoidal_coeff
        part_grad_input = kappa_param*input*sigmoidal_coeff + p

        grad_input = grad_output*part_grad_input
        grad_lambda = grad_output*part_grad_lambda
        grad_kappa = grad_output*part_grad_kappa

        return grad_input, grad_lambda, grad_kappa


class UniRect(nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1,lambda_init=(0.0,1.0),kappa_init=(0.8,1.2),device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        lambda_param = torch.nn.init.uniform_(torch.empty(num_parameters, **factory_kwargs),a=lambda_init[0],b=lambda_init[1])
        kappa_param = torch.nn.init.uniform_(torch.empty(num_parameters, **factory_kwargs),a=kappa_init[0],b=kappa_init[1])

        if num_parameters>1:
            self.lambda_param = nn.Parameter(lambda_param.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
            self.kappa_param = nn.Parameter(kappa_param.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        else:
            self.lambda_param = nn.Parameter(lambda_param)
            self.kappa_param = nn.Parameter(kappa_param)

        self.relu = torch.nn.ReLU()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        lambda_param = self.relu(self.lambda_param) + 1e-8
        out = UniRect_static.apply(input, lambda_param, self.kappa_param)

        return out

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)

    
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
    
class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.bias = Parameter(torch.randn(out_features))

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class CosNorm_Classifier(nn.Module):
    def __init__(self, in_dims, out_dims, scale=16, margin=0.5, init_std=0.001,learnable=False):
        super(CosNorm_Classifier, self).__init__()
        self.in_features = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.learnable = learnable
        if self.learnable is True:
            self.scale = Parameter(torch.FloatTensor(1).cuda())

        self.margin = margin
        self.weight = Parameter(torch.Tensor(out_dims, in_dims).cuda())
        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.learnable is True:
            self.scale.data.fill_(5)

    def forward(self, input, *args):
        norm_x = torch.norm(input.clone(), 2, 1, keepdim=True)
        
        ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        # ex = input/ (1 + norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        
        if self.learnable is True:
            return torch.mm((self.scale**2) * ex, ew.t())
        else:
            return torch.mm(self.scale * ex, ew.t())

class InvCosNorm_Classifier(nn.Module):
    def __init__(self, in_dims, out_dims, scale=1/16, margin=0.5, init_std=0.001,learnable=False):
        super(InvCosNorm_Classifier, self).__init__()
        self.in_features = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.learnable = learnable
        if self.learnable is True:
            self.scale = Parameter(torch.FloatTensor(1).cuda())

        self.margin = margin
        self.weight = Parameter(torch.Tensor(out_dims, in_dims).cuda())
        self.bias = Parameter(torch.Tensor(out_dims).cuda())
        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.learnable is True:
            self.scale.data.fill_(5)

    def forward(self, input, *args):
        norm_x = torch.norm(input.clone(), 2, 1, keepdim=True)
        
        ex = input* (1 + norm_x)
        ew = self.weight * torch.norm(self.weight, 2, 1, keepdim=True)
        
        if self.learnable is True:
            return torch.mm((self.scale**2) * ex, ew.t()) 
        else:
            return torch.mm(self.scale * ex, ew.t()) + self.bias
        
class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=4,use_gumbel=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.use_gumbel = use_gumbel
        if self.use_gumbel is True:
            self.unact = Unified()
            self.norm = nn.LayerNorm(c)
            self.excitation = nn.Sequential(
                nn.Linear(c, c // r, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(c // r, c, bias=False),
                nn.Dropout(p=0.1))
        else:
            self.excitation = nn.Sequential(
                nn.Linear(c, c // r, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(c // r, c, bias=False))
            
        

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        if self.use_gumbel is True:
            y= self.norm(y)
        y = self.excitation(y).view(bs, c, 1, 1)
        if self.use_gumbel is True:
            y=self.unact(y)
        else:
            y=y.sigmoid()
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))        
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


class BasicBlockGumbel(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.unirect1 = UniRect()
        self.unirect2 = UniRect()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.unirect1(self.bn1(self.conv1(x)))        
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.unirect2(out)
        
        return out

class Se_Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(Se_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SE_Block(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class Se_Block_Gumbel(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(Se_Block_Gumbel, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SE_Block(planes,use_gumbel=True)
        self.shortcut = nn.Sequential()
        self.unirect1 = UniRect()
        self.unirect2 = UniRect()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.unirect1(self.bn1(self.conv1(x)))        
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.unirect2(out)

        return out

    
class Cb_Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(Cb_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.cb = cbam.CBAM(planes,reduction_ratio=4)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cb(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class Cb_Block_Gumbel(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(Cb_Block_Gumbel, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.cb = cbam.CBAM(planes,reduction_ratio=4,use_gumbel=True,use_gumbel_cb=True)
        self.unirect1 = UniRect()
        self.unirect2 = UniRect()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.unirect1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cb(out)
        out += self.shortcut(x)
        out = self.unirect2(out)
        return out
    
class Cb_Block_GumbelSigmoid(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(Cb_Block_GumbelSigmoid, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.cb = cbam.CBAM(planes,reduction_ratio=4,use_gumbel=True,use_gumbel_cb=False)
        self.unirect1 = UniRect()
        self.unirect2 = UniRect()


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.unirect1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cb(out)
        out += self.shortcut(x)
        out = self.unirect2(out)
        return out

class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=None):
        super(ResNet_s, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if use_norm=='norm':
            self.linear = NormedLinear(64, num_classes)
        elif use_norm=='cosine':
            self.linear = CosNorm_Classifier(64, num_classes)
        elif use_norm=='lr_cosine':
            self.linear = CosNorm_Classifier(64, num_classes,learnable=True)
        else:
            self.linear = nn.Linear(64, num_classes)
        
        if (block.__name__.endswith('Gumbel')) or (block.__name__.endswith('GumbelSigmoid')) is True:
            self.unirect = UniRect()
        else:
            self.unirect = nn.ReLU(inplace=True)
        
        self.apply(_weights_init)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
        out = self.unirect(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
            
        

class ContrastiveLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(ContrastiveLayer, self).__init__()
        # self.contrast_reduce = nn.Sequential(nn.Linear(in_features, in_features//2),nn.Linear(in_features//2, in_features//4))
        self.contrast_reduce =nn.Linear(in_features, in_features)
        # self.contrast_expand = CosNorm_Classifier(in_features//2,out_features,scale=1)
        # self.contrast_expand = NormedLinear(in_features,out_features)
        self.contrast_expand = nn.Linear(in_features,out_features)

    def forward(self, x):
        out = self.contrast_reduce(x)
        out= F.relu(out)
        out = self.contrast_expand(out)
        return out

class ResNet_Contrastive(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=None):
        super(ResNet_Contrastive, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        if use_norm=='norm':
            self.linear = NormedLinear(64, num_classes)
        elif use_norm=='cosine':
            self.linear = CosNorm_Classifier(64, num_classes)
        else:
            self.linear = nn.Linear(64, num_classes)
        self.contrastive = ContrastiveLayer(64,128)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)

        logits = self.linear(out)
        features = self.contrastive(out)
        return logits,features


class ResNet_TwoBranch(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=None):
        super(ResNet_TwoBranch, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        if use_norm=='norm':
            self.linear1 = NormedLinear(64, num_classes)
            self.linear2 = NormedLinear(64, num_classes)
        elif use_norm=='cosine':
            self.linear1 = CosNorm_Classifier(64, num_classes)
            self.linear2 = CosNorm_Classifier(64, num_classes)
        else:
            self.linear1 = nn.Linear(64, num_classes)
            self.linear2 = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)

        logits1 = self.linear1(out)
        logits2 = self.linear2(out)

        return logits1,logits2


def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])


def resnet32(num_classes=10, use_norm=None,use_gumbel=False,use_gumbel_cb=False):
    if use_gumbel is False:
        return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)
    else:
        return ResNet_s(BasicBlockGumbel, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)

def se_resnet32(num_classes=10, use_norm=None,use_gumbel=False,use_gumbel_cb=False):
    if use_gumbel is False:
        return ResNet_s(Se_Block, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)
    else:
        return ResNet_s(Se_Block_Gumbel, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)
    
def cb_resnet32(num_classes=10, use_norm=None,use_gumbel=False,use_gumbel_cb=False):
    if use_gumbel is False:
        return ResNet_s(Cb_Block, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)
    else:
        if use_gumbel_cb is True:
            return ResNet_s(Cb_Block_Gumbel, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)
        else:
            return ResNet_s(Cb_Block_GumbelSigmoid, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)

def con_resnet32(num_classes=10, use_norm=None):
    return ResNet_Contrastive(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)

def tb_resnet32(num_classes=10, use_norm=None):
    return ResNet_TwoBranch(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)

def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])



def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()