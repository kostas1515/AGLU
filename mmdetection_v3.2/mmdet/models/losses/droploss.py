import torch
import torch.nn as nn
import torch.nn.functional as F
from .accuracy import accuracy
from mmdet.registry import MODELS

def get_image_count_frequency(version="v0_5"):
    if version == "v0_5":
        from mmdet.utils.lvis_v0_5_categories import get_image_count_frequency
        return get_image_count_frequency()
    elif version == "v1":
        from mmdet.utils.lvis_v1_0_categories import get_image_count_frequency
        return get_image_count_frequency()
    elif version == "openimage":
        from mmdet.utils.openimage_categories import get_instance_count
        return get_instance_count()
    else:
        raise KeyError(f"version {version} is not supported")
        

def logsumexp(x):
    alpha=torch.exp(x)
    return alpha+torch.log(1.0-torch.exp(-alpha))

@MODELS.register_module()
class DropLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 lambda_= 0.0011,
                 version="v1",
                 use_classif='gumbel',
                 num_classes=1203):
        super(DropLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.lambda_ = lambda_
        self.version = version
        self.freq_info = torch.cuda.FloatTensor(get_image_count_frequency(version))
        self.use_classif=use_classif
        self.num_classes=num_classes
        self.custom_cls_channels = True
        self.custom_activation = True
        self.custom_accuracy = True
        

        num_class_included = torch.sum(self.freq_info < self.lambda_)
        print(f"set up DropLoss (version {version}), {num_class_included} classes included.")
        
    def get_cls_channels(self, num_classes):
        """Get custom classification channels.

        Args:
            num_classes (int): The number of classes.

        Returns:
            int: The custom classification channels.
        """
        assert num_classes == self.num_classes
        return num_classes
    
    def get_accuracy(self, cls_score, labels):
        
        pos_inds = labels < self.num_classes
        acc_classes = accuracy(cls_score[pos_inds], labels[pos_inds])
        acc = dict()
        acc['acc_classes'] = acc_classes
        return acc
    
    def get_activation(self, cls_score):
        """Get custom activation of cls_score.

        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C).

        Returns:
            torch.Tensor: The custom activation of cls_score with shape
                 (N, C).
        """
        
        if self.use_classif=='gumbel':
            scores = 1/(torch.exp(torch.exp(-cls_score)))
        else:
            scores=torch.sigmoid(cls_score)
        
        dummpy_prob = scores.new_zeros((scores.size(0), 1))
        scores = torch.cat([scores, dummpy_prob], dim=1)
        
        return scores
        

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        
        
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.n_c]

        target = expand_label(cls_score, label)
        drop_w = 1 - self.threshold_func() * (1 - target)
        
        if self.use_classif =='gumbel':
            # cls_score=torch.clamp(cls_score,min=-4,max=10)
            # pestim= 1/(torch.exp(torch.exp(-(cls_score))))
            # cls_loss = F.binary_cross_entropy(pestim, target,reduction='none')
            cls_score=torch.clamp(cls_score,min=-10,max=10)
            cls_loss=torch.exp(-cls_score)*target.float() +(target.float()-1.0)*(logsumexp(-cls_score)-torch.exp(-cls_score))
        else:
            cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,reduction='none')


        cls_loss = torch.sum(cls_loss * drop_w) / self.n_i

        return self.loss_weight * cls_loss
    

    def exclude_func_and_ratio(self):
        
        # instance-level weight
        bg_ind = self.n_c
        weight = (self.gt_classes != bg_ind)

        gt_classes    = self.gt_classes[weight]
        exclude_ratio = torch.mean((self.freq_info[gt_classes] < self.lambda_).float())

        weight = weight.float().view(self.n_i, 1).expand(self.n_i, self.n_c)

        return weight, exclude_ratio

    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        weight[self.freq_info < self.lambda_] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)


        fg, ratio = self.exclude_func_and_ratio()
        bg = 1 - fg
        random = torch.rand_like(bg) * bg

        random = torch.where(random>ratio, torch.ones_like(random), torch.zeros_like(random))
        
        weight = (random + fg) * weight

        return weight