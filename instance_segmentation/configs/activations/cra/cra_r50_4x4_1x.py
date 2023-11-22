_base_ = [
    '../../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

data = dict(train=dict(oversample_thr=0.0))


model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="ClassRelativeActivations"),
                                         init_cfg = dict(type='Constant',val=0.001, bias=-3.0, override=dict(name='fc_cls')))))

work_dir='./experiments/cra_r50_4x4_1x/'


