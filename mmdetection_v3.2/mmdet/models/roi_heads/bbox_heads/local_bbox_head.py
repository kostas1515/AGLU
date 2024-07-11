# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead
import torch.nn.functional as F
from mmdet.models.layers import multiclass_nms
import pandas as pd
import numpy as np
import torch
from scipy.special import ndtri
from torch import Tensor
from typing import Optional
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.models.utils import empty_instances
from mmdet.structures.bbox import get_box_tensor, scale_boxes





@MODELS.register_module()
class ConvFCLocalBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 variant='raw',
                 fractal_l=4.0,
                 *args,
                 **kwargs):
        super(ConvFCLocalBBoxHead,
              self).__init__(num_shared_convs, num_shared_fcs, num_cls_convs,
                             num_cls_fcs, num_reg_convs, num_reg_fcs,
                             conv_out_channels, fc_out_channels, conv_cfg,
                             norm_cfg, init_cfg, *args, **kwargs)
        self.fractal_l = fractal_l
        try:
            # self.wy = self.get_local_weightsv2('./lvis_files/idf_v3det_train.csv',self.num_classes,variant)
            # self.fractal_weights = pd.read_csv('./lvis_files/v3det_train_fractal_dim.csv')['fractal_dimension'].values.tolist()+[1.0]
            self.wy = self.get_local_weightsv2('./lvis_files/open_images_idf.csv',self.num_classes,variant)
            self.fractal_weights = pd.read_csv('./lvis_files/fractal_dims_open_images.csv')['fractal_dimension'].values.tolist()+[1.0]
           
        except FileNotFoundError:
            self.wy = self.get_local_weightsv2('../lvis_files/open_images_idf.csv',self.num_classes,variant)
            self.fractal_weights = pd.read_csv('../lvis_files/fractal_dims_open_images.csv')['fractal_dimension'].values.tolist()+[1.0]

        self.fractal_weights = torch.tensor(self.fractal_weights,device='cuda')

        self.wy[torch.isinf(self.wy)]= 0.0
        self.wy= 0.0    


        

    def get_local_weightsv2(self,lvis_file,num_categories,variant):
        
        fg_weights =pd.read_csv(lvis_file)[variant].values[1:]
        print(fg_weights)
        if variant.endswith('_obj'): # this accomodates the smooth variant that is dependent on the frequency, it does not matter for other variants
            freqs = np.ones(num_categories)*1000 
        else:
            freqs = np.ones(num_categories)*300
            
        
        if variant.startswith('raw'):
            ptarget = -np.log(freqs.sum()/freqs)
        elif variant.startswith('smooth'):
            ptarget = -np.log((freqs.sum()+1)/(freqs+1))+1
        elif variant.startswith('prob'):
            ptarget = -np.log((freqs.sum()-freqs)/freqs)
        elif variant.startswith('normit'):
            ptarget = ndtri(freqs/freqs.sum())
        elif variant.startswith('gombit'):
            ptarget = np.log(-np.log(1-(freqs/freqs.sum())))
        elif variant.startswith('base2'):
            ptarget = -np.log2(freqs.sum()/freqs)
        elif variant.startswith('base10'):
            ptarget = -np.log10(freqs.sum()/freqs)
        
        fg_weights = fg_weights + ptarget
        fg_weights = fg_weights.tolist()+[0.0]
        # fg_weights = fg_weights.tolist()

        return torch.tensor(fg_weights,device='cuda',dtype=torch.float).unsqueeze(0)
  

    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]
        # local calibration
        weights = self.wy


        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score+weights)
            # fractal_weights = (self.fractal_weights**self.fractal_l)
            # fractal_weights= fractal_weights/fractal_weights.sum()
            # fractal_prob = -torch.log10(fractal_weights)+torch.log10(torch.tensor(1/self.num_classes)).item()
            # scores = self.loss_cls.get_activation(cls_score) * (self.loss_cls.get_activation(cls_score+weights+fractal_prob))
        else:
            scores = F.softmax(
                cls_score+weights, dim=-1) if cls_score is not None else None

        scores = scores/(self.fractal_weights**self.fractal_l)
        scores /= scores.sum(dim=1, keepdim=True)

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results
       
@MODELS.register_module()
class Shared2FCLocalBBoxHead(ConvFCLocalBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCLocalBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        

        
        

    
    