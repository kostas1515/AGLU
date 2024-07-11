_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py', '../_base_/datasets/lvis_v1_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    backbone=dict(
        type='SE_ResNet',
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            './pretrained/imagenet_full_se_r101_cosine_scheduler_mixup_aug_unirect_relinse_softplus_99'
        ),
        use_gumbel=True),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1203,
            loss_cls=dict(
                type='DropLoss',
                use_sigmoid=True,
                loss_weight=1.0,
                lambda_=0.0011,
                version='v1',
                use_classif='gumbel'),
            init_cfg=dict(
                type='Constant',
                val=0.001,
                bias=-2,
                override=dict(name='fc_cls'))),
        mask_head=dict(
            type='FCNMaskHead',
            predictor_cfg=dict(type='NormedConv2d', tempearture=20),
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1203,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.3),
            max_per_img=300,
            mask_thr_binary=0.4,
            perclass_nms=True)))
# dataset settings
fp16 = dict(loss_scale=512.) 
evaluation = dict(metric=['bbox', 'segm'], interval=24)
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=4)

# work_dir = './experiments/mask_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x/'
work_dir = './experiments/mask_rcnn_r101_fpn_8x2_sample1e-3_mstrain_lvis_2x_repro/'
