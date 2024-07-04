import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Unified(nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1,device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        lambda_param = torch.nn.init.uniform_(torch.empty(num_parameters, **factory_kwargs))
        kappa_param = torch.nn.init.uniform_(torch.empty(num_parameters, **factory_kwargs))
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
    

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg','max'],use_gumbel=False):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.norm = nn.LayerNorm(gate_channels)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
            nn.Dropout(p=0.1)
            )
        self.pool_types = pool_types
        self.use_gumbel=use_gumbel
        if self.use_gumbel is True:
            self.uniact=Unified()
        
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                avg_pool = self.norm(avg_pool.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                max_pool = self.norm(max_pool.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                lp_pool = self.norm(lp_pool.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x).unsqueeze(-1).unsqueeze(-1)
                lse_pool = self.norm(lse_pool)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        
        scale = channel_att_sum.unsqueeze(2).unsqueeze(3).expand_as(x)
        if self.use_gumbel is True:
#             scale=torch.exp(-torch.exp(-torch.clamp(scale,min=-4.0,max=10.0)))
            scale =  self.uniact(scale)
        else:
            scale=scale.sigmoid()
            
#         scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self,use_gumbel=False):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.use_gumbel=use_gumbel
        
        if self.use_gumbel is True:
            self.uniact=Unified()
        
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        if self.use_gumbel is True:
            scale =  self.uniact(x_out)
        else:
            scale=x_out.sigmoid()
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg','max'], no_spatial=False,use_gumbel=False,use_gumbel_cb=False,no_channel=False):
        super(CBAM, self).__init__()
        self.no_channel=no_channel
        if not self.no_channel:
            self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types,use_gumbel=use_gumbel)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(use_gumbel=(use_gumbel_cb))
    def forward(self, x):
        if not self.no_channel:
            x = self.ChannelGate(x)
        if not self.no_spatial:
            x = self.SpatialGate(x)
        return x
