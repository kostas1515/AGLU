import torch
import torch.nn as nn
import numpy as np
from scipy.special import ndtri
import itertools

class BCE(nn.Module):
    def __init__(self,reduction='mean',label_smoothing=0.0,weight=None):
        super(BCE, self).__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing  
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none',weight=weight)
        
    def forward(self, pred, targets):
        nc=pred.shape[-1]
        if (targets.size() == pred.size()) is False:
            y_onehot = torch.cuda.FloatTensor(pred.shape)
            y_onehot.zero_()
            y_onehot.scatter_(1, targets.unsqueeze(1), 1)
            y_onehot_smoothed = y_onehot*(1-self.label_smoothing) + self.label_smoothing/nc
        else:
            y_onehot_smoothed = targets*(1-self.label_smoothing) + self.label_smoothing/nc


        loss = self.loss_fcn(pred,y_onehot_smoothed)
            
        if self.reduction=='mean':
            loss=loss.mean()
        elif self.reduction=='sum':
            loss=loss.sum()/y_onehot_smoothed.sum()
             
        return loss

class IIFLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self,dataset,variant='rel',device='cuda',weight=None,label_smoothing=None):
        super(IIFLoss, self).__init__()
        self.loss_fcn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing,weight=weight)
        self.variant = variant
        freqs = np.array(dataset.get_cls_num_list())
        iif={}
        iif['raw']= np.log(freqs.sum()/freqs)
        iif['smooth'] = np.log((freqs.sum()+1)/(freqs+1))+1
        iif['rel'] = np.log((freqs.sum()-freqs)/freqs)
        
        iif['normit'] = -ndtri(freqs/freqs.sum())
        iif['gombit'] = -np.log(-np.log(1-(freqs/freqs.sum())))
        iif['base2'] = np.log2(freqs.sum()/freqs)
        iif['base10'] = np.log10(freqs.sum()/freqs)
        self.iif = {k: torch.tensor([v],dtype=torch.float).to(device,non_blocking=True) for k, v in iif.items()}
        
    def forward(self, pred, targets=None,infer=False):
    
        if infer is False:
            loss = self.loss_fcn(pred,targets)
            return loss
        else:
            out = (pred+self.iif[self.variant])

            return out