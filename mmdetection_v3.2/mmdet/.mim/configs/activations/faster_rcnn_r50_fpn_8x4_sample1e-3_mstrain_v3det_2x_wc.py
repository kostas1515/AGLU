_base_ = [
    '../v3det/faster_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x.py'
]
# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            cls_predictor_cfg=dict(_delete_=True,
                type='WCLinear',s_trainable=False),
           loss_cls=dict(_delete_=True,
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0))))

work_dir = './experiments/faster_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x_wc/'
# work_dir = './experiments/test/'
