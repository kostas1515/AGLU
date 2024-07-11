#!/bin/bash -i
WORK_DIR="${HOME}/work/konsa15"
${WORK_DIR}/miniconda3/bin/conda init bash
source /home/ma-user/.bashrc

cd ..

export TORCH_CPP_LOG_LEVEL=DEBUG

conda activate torch
./tools/dist_test.sh ./configs/activations/mask_rcnn_r101_fpn_8x2_sample1e-3_mstrain_lvis_2x.py \
 ./experiments/droploss_normed_mask_se_r101_rfs_4x4_2x_gumbel_uni_se_uniact/epoch_24.pth 4

./tools/dist_test.sh ./experiments/coco/se_faster_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x_wc/se_faster_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x_wc_unirect.py \
 ./experiments/coco/se_faster_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x_wc/iter_137520.pth 4

./tools/dist_test.sh ./experiments/coco/se_cascade_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x_wc_unirect/se_cascade_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x_wc_unirect.py \
 ./experiments/coco/se_cascade_rcnn_r50_fpn_8x4_sample1e-3_mstrain_v3det_2x_wc_unirect/iter_137520.pth 4