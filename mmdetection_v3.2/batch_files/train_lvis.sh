#!/bin/bash -i
WORK_DIR="${HOME}/work/konsa15"
${WORK_DIR}/miniconda3/bin/conda init bash
source /home/ma-user/.bashrc

cd ..
conda activate torch && ./tools/dist_train.sh ./configs/activations/mask_rcnn_r50_fpn_8x2_sample1e-3_mstrain_lvis_2x.py 8