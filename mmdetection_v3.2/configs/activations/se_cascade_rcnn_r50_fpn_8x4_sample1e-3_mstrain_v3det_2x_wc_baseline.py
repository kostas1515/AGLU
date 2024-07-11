_base_ = [
    '../v3det/cascade_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x.py'
]
# model settings
model = dict(
    backbone=dict(
        type='SE_ResNet',
        depth=50,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./pretrained/imagenet_full_se_r50_cosine_scheduler_mixup_aug_99.pth'),
        use_gumbel=False),
    roi_head=dict(
        bbox_head=[
        dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=13204,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            cls_predictor_cfg=dict(
                type='WCLinear',s_trainable=False),
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=13204,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1]),
            reg_class_agnostic=True,
            cls_predictor_cfg=dict(
                type='WCLinear',s_trainable=False),
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=13204,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.033, 0.033, 0.067, 0.067]),
            reg_class_agnostic=True,
            cls_predictor_cfg=dict(
                type='WCLinear',s_trainable=False),
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))
    ]))


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=1e-4 * 1, weight_decay=0.1),
    clip_grad=dict(max_norm=35, norm_type=2))

fp16 = dict(loss_scale=512.) 

work_dir = "./experiments/coco/se_cascade_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x_wc_baseline"
# work_dir = "./experiments/coco/test"
