<h1> Gumbel Optimised Loss for Long-tailed Instance Segmentation </h1>

This is the official implementation of Gumbel Optimised Loss for Long-tailed Instance Segmentation for ECCV2022 accepted paper.

Gumbel Activation using (M)ask-RCNN, (R)esnet,Resne(X)t, (C)ascade Mask-RCNN and (H)ybrid Task Cascade.
<img src="./figures/ap_maskrcnn.jpg"
     alt="Performance of Gumbel activation"
     style="float: left; margin-right: 10px;"
/>
<h1> Gumbel Cross Entropy (simplified)</h1>

```
def gumbel_cross_entropy(pred,
                         label,reduction):
    """Calculate the Gumbel CrossEntropy loss.
    Args:
        pred (torch.Tensor): The prediction.
        label (torch.Tensor): one-hot encoded
    Returns:
        torch.Tensor: The calculated loss.
    """
    pred=torch.clamp(pred,min=-4,max=10)
    pestim= 1/(torch.exp(torch.exp(-(pred))))
    loss = F.binary_cross_entropy(
        pestim, label.float(), reduction=reduction)
    loss=torch.clamp(loss,min=0,max=20)

    return loss
```

<h1> Tested with </h1>
<div>
 <ul>
  <li>python==3.8.12</li>
  <li>torch==1.7.1</li>
  <li>torchvision==0.8.2</li>
  <li>mmdet==2.21.0</li>
  <li>lvis</li>
  <li>Tested on CUDA 10.2 and RHEL 8 system</li>
</ul> 
</div>


<h1> Getting Started </h1>
Create a virtual environment

```
conda create --name mmdet pytorch=1.7.1 -y
conda activate mmdet
```

1. Install dependency packages
```
conda install torchvision -y
conda install pandas scipy -y
conda install opencv -y
```

1. Install MMDetection
```
pip install openmim
mim install mmdet==2.21.0
```
3. Clone this repo
```
git clone https://github.com/kostas1515/GOL.git
cd GOL
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

5. modify mmdetection/configs/_base_/datasets/lvis_v1_instance.py and make sure data_root variable points to the above data directory, e.g.,
data_root = '<user_path>'

<h1>Training</h1>
To Train on multiple GPUs use <i>tools/dist_train.sh</i> to launch training on multiple GPUs:

```
./tools/dist_train.sh ./configs/<experiment>/<variant.py> <#GPUs>
```

E.g: To train GOL on 4 GPUs use:
```
./tools/dist_train.sh ./configs/gol/droploss_normed_mask_r50_rfs_4x4_2x_gumbel.py 4
```
<h1>Testing</h1>

To test GOL:
```
./tools/dist_test.sh ./experiments/droploss_normed_mask_rcnn_r50_rfs_4x4_2x_gumbel/droploss_normed_mask_r50_rfs_4x4_2x_gumbel.py ./experiments/droploss_normed_mask_r50_rfs_4x4_2x_gumbel/latest.pth 4 --eval bbox segm
```


<h1>Reproduce</h1>
To reproduce the results on the the paper with Sigmoid, Softmax and Gumbel activation run:

```
./tools/dist_train.sh ./configs/activations/r50_4x4_1x.py <#GPUs>
./tools/dist_train.sh ./configs/activations/r50_4x4_1x_softmax.py <#GPUs>
./tools/dist_train.sh ./configs/activations/gumbel/gumbel_r50_4x4_1x.py <#GPUs>
```
It will give a Table similar to this:
<table style="float: center; margin-right: 10px;">
    <tr>
        <th>Method</th>
        <th>AP</th>
        <th>AP<sup>r</sup></th>
        <th>AP<sup>c</sup></th>
        <th>AP<sup>f</sup></th>
        <th>AP<sup>b</sup></th>
    </tr>
    <tr>
        <td>Sigmoid</td>
        <td>16.4</td>
        <td>0.8</td>
        <td>12.7</td>
        <td>27.3</td>
        <td>17.2</td>
    </tr>
    <tr>
        <td>Softmax</td>
        <td>15.2</td>
        <td>0.0</td>
        <td>10.6</td>
        <td>26.9</td>
        <td>16.1</td>
    </tr>
    <tr>
        <td>Gumbel</td>
        <td><b>19.0</b></td>
        <td><b>4.9</b></td>
        <td><b>16.8</b></td>
        <td><b>27.6</b></td>
        <td><b>19.1</b></td>
    </tr>

</table>
    
<h1>Pretrained Models</h1>
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
        <td>GOL_r50_v0.5</td>
        <td>29.5</td>
        <td>22.5</td>
        <td>31.3</td>
        <td>30.1</td>
        <td>28.2</td>
        <td><a href="https://www.dropbox.com/s/pl2t9aug7rrwuja/epoch_24.pth?dl=0">weights</a></td>
        <td><a href="https://www.dropbox.com/s/6tc73ke3hq8zqzc/20220524_141924.log?dl=0">log</a>|<a href="https://www.dropbox.com/s/lqb2tbo9771tu04/droploss_normed_mask_r50_lvis05_rfs_4x4_2x_gumbel.py?dl=0">config</a></td>
    </tr>
    <tr>
        <td>GOL_r50_v1</td>
        <td>27.7</td>
        <td>21.4</td>
        <td>27.7</td>
        <td>30.4</td>
        <td>27.5</td>
        <td><a href="https://www.dropbox.com/s/caav66oardal9ny/epoch_24.pth?dl=0">weights</a></td>
        <td><a href="https://www.dropbox.com/s/ei31bb2supyn6ef/20220711_133821.log?dl=0">log</a>|<a href="https://www.dropbox.com/s/64vkqc83m2etx6l/droploss_normed_mask_r50_rfs_4x4_2x_gumbel.py?dl=0">config</a></td>
    </tr>
    <tr>
        <td>GOL_r101_v1</td>
        <td>29.0</td>
        <td>22.8</td>
        <td>29.0</td>
        <td>31.7</td>
        <td>29.2</td>
        <td><a href="https://www.dropbox.com/s/l76cge8hbb4s2e9/epoch_24.pth?dl=0">weights</a></td>
        <td><a href="https://www.dropbox.com/s/o92neoc1ogopokg/20220711_074416.log?dl=0">log</a>|<a href="https://www.dropbox.com/s/n2325d7q534x6g8/droploss_normed_mask_r101_rfs_4x4_2x_gumbel.py?dl=0">config</a></td>
    </tr>

</table>

     
<h1> Acknowledgements </h1>
     This code uses the <a href='https://github.com/open-mmlab/mmdetection'>mmdet</a> framework. It also uses <a href='https://github.com/tztztztztz/eqlv2'>EQLv2</a> and <a href='https://github.com/timy90022/DropLoss'>DropLoss</a>. Thank you for your wonderfull work! 
     

     
