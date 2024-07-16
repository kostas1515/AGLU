<h1> Tested with </h1>
<div>
 <ul>
  <li>python==3.8.12</li>
  <li>torch==2.3.0</li>
  <li>mmdet==3.2.0</li>
  <li>lvis</li>
  <li>Tested on CUDA 12.1 and Linux x86_64 system</li>
</ul> 
</div>


<h1> Getting Started </h1>
Create a virtual environment

```
conda create --name <your_env> 
conda activate <your_env>
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

1. Install dependency packages
```
conda install pandas scipy
pip install opencv-python-headless
pip install lvis
```

2. Install mmcv
```
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
```
3. Install MMdet
```
git clone https://github.com/kostas1515/AGLU.git
cd mmdet
pip install -v -e .
```
4. Create data directory, download COCO 2017 datasets at https://cocodataset.org/#download (2017 Train images [118K/18GB], 2017 Val images [5K/1GB], 2017 Train/Val annotations [241MB]) and extract the zip files:
```
mkdir data
cd data
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

#download and unzip LVIS annotations
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip

```

modify mmdetection/configs/_base_/datasets/lvis_v1_instance.py and make sure data_root variable points to the above data directory, e.g.,
data_root = '<user_path>'. For V3DET dataset, download the images from https://github.com/V3Det/V3Det.

<h1>Training</h1>
To Train on multiple GPUs use <i>tools/dist_train.sh</i> to launch training on multiple GPUs:

```
./tools/dist_train.sh ./configs/<experiment>/<variant.py> <#GPUs>
```

E.g: To train MaskRCNN SE-R50+AGLU+GOL on LVIS using 8 GPUs use:
```
./tools/dist_train.sh ./configs/activatios/mask_rcnn_r50_fpn_8x2_sample1e-3_mstrain_lvis_2x.py 8
```

Be sure to include the APA-AGLU-SE-R50 ImageNet1K pretrained backbone as initialisation. These models are provided in the classification folder. 

<h1>Testing</h1>

To test MaskRCNN SE-R50-AGLU+GOL on LVIS use:
```
./tools/dist_test.sh ./experiments/droploss_normed_mask_se_r50_rfs_4x4_2x_gumbel_uni_se_uniact/droploss_normed_mask_se_r50_rfs_4x4_2x_gumbel_uni_se_uniact.py ./experiments/droploss_normed_mask_se_r50_rfs_4x4_2x_gumbel_uni_se_uniact/epoch_24.pth 8
```

    
<h3>Pretrained Models on LVIS</h3>
<table style="float: center; margin-right: 10px;">
    <tr>
        <th>Method</th>
        <th>AP</th>
        <th>AP<sup>r</sup></th>
        <th>AP<sup>c</sup></th>
        <th>AP<sup>f</sup></th>
        <th>AP<sup>b</sup></th>
        <th>Model</th>
        <th>Output</th>
    </tr>
    <tr>
        <td>SE-MaskRCNN-R50-AGLU-APA-GOL</td>
        <td>29.2</td>
        <td>21.8</td>
        <td>29.8</td>
        <td>31.9</td>
        <td>29.1</td>
        <td><a href="https://drive.usercontent.google.com/download?id=1clsDFeWA8cI6JeeMNSLFuV_bVASKG5w1&export=download">weights</a></td>
        <td><a href="https://drive.google.com/file/d/1HlKr3GA8IDKiX351MMiXARLei6JjOOBC/view">log</a>|<a href="https://drive.google.com/file/d/1LDoqJIXpKtmR_n0QWeYMi7pU3Z593jNY/view">config</a></td>
    </tr>
    <tr>
        <td>SE-MaskRCNN-R101-AGLU-APA-GOL</td>
        <td>30.7</td>
        <td>23.6</td>
        <td>31.3</td>
        <td>33.1</td>
        <td>31.1</td>
        <td><a href="https://drive.usercontent.google.com/download?id=1WowqsZhSZZ_z-oBkcR3bLSEPLyc97llD&export=download">weights</a></td>
        <td><a href="https://drive.google.com/file/d/1kTTxzTPMlFcXYNMvHtY7smR5Nzww_RoX/view">log</a>|<a href="https://drive.google.com/file/d/1jGt8yEsYyAzL7s7pY8k1Pdg8IhJ2zevY/view">config</a></td>
    </tr>
</table>


<h3>Pretrained Models on V3DET</h3>
<table style="float: center; margin-right: 10px;">
    <tr>
        <th>Method</th>
        <th>AP<sup>b</sup></th>
        <th>Model</th>
        <th>Output</th>
    </tr>
    <tr>
        <td>SE-FasterRCNN-R50-AGLU-APA</td>
        <td>29.9</td>
        <td><a href="https://drive.usercontent.google.com/download?id=1kSZkewkLNvpRcIE9f2fHVDvvn4xNCiI-&export=download">weights</a></td>
        <td><a href="https://drive.google.com/file/d/1BmDTFyMDfzO2uU7Je1s2x-28TBPv2eQR/view">log</a>|<a href="https://drive.google.com/file/d/1SyDZrqNKL3kC2EQDBAG-XLzO3NGqnOMF/view">config</a></td>
    </tr>
    <tr>
        <td>SE-CascadeRCNN-R50-AGLU-APA</td>
        <td>35.4</td>
        <td><a href="https://drive.usercontent.google.com/download?id=1dhF2N-4ndpjFt46MA5hsnXx1zoUGks6u&export=download">weights</a></td>
        <td>n/a|<a href="https://drive.google.com/file/d/1l1iA1fBN2fGZV9GxK6k_vLLgk5UJ_IFX/view">config</a></td>
    </tr>
</table>
