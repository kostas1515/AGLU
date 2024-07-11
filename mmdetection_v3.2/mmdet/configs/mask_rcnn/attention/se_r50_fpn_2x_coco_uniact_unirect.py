_base_ = '../mask_rcnn_r50_fpn_2x_coco.py'

model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='SE_ResNet',
        depth=50,
        use_gumbel=True))

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)


# work_dir = "./experiments/coco/se_r50_fpn_2x_coco_uniact_unirect"

work_dir = "./experiments/coco/test"

