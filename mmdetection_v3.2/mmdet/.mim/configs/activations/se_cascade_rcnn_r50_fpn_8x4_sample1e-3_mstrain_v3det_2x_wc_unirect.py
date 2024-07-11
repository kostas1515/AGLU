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
            checkpoint='./pretrained/imagenet_full_se_r50_cosine_scheduler_mixup_aug_unirect_relinse_99.pth'),
        use_gumbel=True),
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


# paramwise_cfg=dict(custom_keys={'bacbone.relu.lambda_param': dict(decay_mult=0.0), 'bacbone.relu.kappa_param': dict(decay_mult=0.0), 'bacbone.layer1.0.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer1.0.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer1.0.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer1.0.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer1.0.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer1.0.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer1.0.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer1.0.se.uniact.kappa_param': dict(decay_mult=0.0), 'bacbone.layer1.1.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer1.1.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer1.1.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer1.1.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer1.1.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer1.1.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer1.1.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer1.1.se.uniact.kappa_param': dict(decay_mult=0.0), 'bacbone.layer1.2.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer1.2.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer1.2.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer1.2.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer1.2.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer1.2.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer1.2.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer1.2.se.uniact.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.0.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.0.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.0.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.0.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.0.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.0.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.0.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.0.se.uniact.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.1.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.1.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.1.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.1.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.1.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.1.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.1.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.1.se.uniact.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.2.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.2.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.2.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.2.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.2.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.2.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.2.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.2.se.uniact.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.3.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.3.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.3.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.3.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.3.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.3.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer2.3.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer2.3.se.uniact.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.0.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.0.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.0.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.0.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.0.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.0.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.0.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.0.se.uniact.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.1.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.1.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.1.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.1.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.1.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.1.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.1.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.1.se.uniact.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.2.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.2.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.2.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.2.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.2.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.2.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.2.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.2.se.uniact.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.3.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.3.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.3.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.3.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.3.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.3.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.3.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.3.se.uniact.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.4.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.4.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.4.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.4.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.4.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.4.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.4.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.4.se.uniact.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.5.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.5.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.5.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.5.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.5.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.5.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer3.5.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer3.5.se.uniact.kappa_param': dict(decay_mult=0.0), 'bacbone.layer4.0.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer4.0.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer4.0.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer4.0.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer4.0.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer4.0.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer4.0.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer4.0.se.uniact.kappa_param': dict(decay_mult=0.0), 'bacbone.layer4.1.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer4.1.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer4.1.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer4.1.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer4.1.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer4.1.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer4.1.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer4.1.se.uniact.kappa_param': dict(decay_mult=0.0), 'bacbone.layer4.2.unirect1.lambda_param': dict(decay_mult=0.0), 'bacbone.layer4.2.unirect1.kappa_param': dict(decay_mult=0.0), 'bacbone.layer4.2.unirect2.lambda_param': dict(decay_mult=0.0), 'bacbone.layer4.2.unirect2.kappa_param': dict(decay_mult=0.0), 'bacbone.layer4.2.unirect3.lambda_param': dict(decay_mult=0.0), 'bacbone.layer4.2.unirect3.kappa_param': dict(decay_mult=0.0), 'bacbone.layer4.2.se.uniact.lambda_param': dict(decay_mult=0.0), 'bacbone.layer4.2.se.uniact.kappa_param': dict(decay_mult=0.0)})

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=1e-4 * 1, weight_decay=0.1),
    clip_grad=dict(max_norm=35, norm_type=2))

fp16 = dict(loss_scale=512.) 

work_dir = "./experiments/coco/se_cascade_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x_wc_unirect"
# work_dir = "./experiments/coco/test"
