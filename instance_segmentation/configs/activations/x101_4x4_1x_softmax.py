_base_ = [
    '../lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1.py'
]

# data = dict(train=dict(oversample_thr=0.0))
data = dict(train=dict(oversample_thr=0.0),samples_per_gpu=1)

model = dict(roi_head=dict(bbox_head=dict(loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False),
                                         init_cfg = dict(type='Constant',val=0.001, bias=-6.5, override=dict(name='fc_cls')))))

work_dir='./experiments/x101_4x4_1x_softmax/'
# work_dir='./experiments/test/'

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11])

fp16 = dict(loss_scale=512.) 

