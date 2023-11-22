_base_ = [
    '../../lvis/cascade_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

data = dict(train=dict(oversample_thr=0.0))

model = dict(roi_head=dict(bbox_head=[dict(
    type='Shared2FCBBoxHead',
    in_channels=256,
    fc_out_channels=1024,
    roi_feat_size=7,
    num_classes=1203,
    bbox_coder=dict(
        type='DeltaXYWHBBoxCoder',
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2]),
    reg_class_agnostic=True,
    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True),
                    init_cfg = dict(type='Constant',val=0.001, 
                    bias=-6.5, override=dict(name='fc_cls')),
    loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                    loss_weight=1.0)),
dict(
    type='Shared2FCBBoxHead',
    in_channels=256,
    fc_out_channels=1024,
    roi_feat_size=7,
    num_classes=1203,
    bbox_coder=dict(
        type='DeltaXYWHBBoxCoder',
        target_means=[0., 0., 0., 0.],
        target_stds=[0.05, 0.05, 0.1, 0.1]),
    reg_class_agnostic=True,
    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True),
                    init_cfg = dict(type='Constant',val=0.001, 
                    bias=-6.5, override=dict(name='fc_cls')),
    loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                    loss_weight=1.0)),
dict(
    type='Shared2FCBBoxHead',
    in_channels=256,
    fc_out_channels=1024,
    roi_feat_size=7,
    num_classes=1203,
    bbox_coder=dict(
        type='DeltaXYWHBBoxCoder',
        target_means=[0., 0., 0., 0.],
        target_stds=[0.033, 0.033, 0.067, 0.067]),
    reg_class_agnostic=True,
    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True),
                    init_cfg = dict(type='Constant',val=0.001, 
                    bias=-6.5, override=dict(name='fc_cls')),
    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
]))

work_dir='./experiments/cascade_r50_4x4_1x/'