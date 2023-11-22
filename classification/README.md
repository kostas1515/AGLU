# Image classification reference training and testing scripts 

This folder contains reference training and testing scripts for image classification.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

## Basic Usage


| Parameter                | value  | description  |
| ------------------------ | ------ |--------------|
| `--model`                |`se_resnet32,se_resnet50,cb_resnet50,simple_vit`|model name, it can be resnet, se_resnet or simple_vit. Check initialise_model.py fyi |
| `--dset_name`            |`cifar{100}, inat18, imagenet_lt, places_lt`|dataset name |
| `--data_path`            |`../../../datasets/`| data path, please change hardcoded paths |
| `--classif_norm`         |`cosine,lr_cosine`| classifier layr, it can be Linear, Cosine, or Cosine with Learnable weight, check resnet_cifar.py fyi |
| `--imb_type`             | `exp`  | Exponential imbalance. Applicable for Cifar Only |
| `--imb_factor`           | `0.01` | Imbalance Factor. Applicable for Cifar Only |
| `--auto-augment`         | `cifar10`|  Augmentations as defined in pytorch reference scripts|
| `--criterion`            | `ce,iif`| When iif, is used then PC_softmax is applied during inference.|
| `--use_gumbel_se`        | `store_true`|It will use Gumbel Channel Attention.|
| `--use_gumbel_cb`        | `store_true`|It will use Gumbel Spatial Attention.|
| `--attn`                 | `softmax,gumbel`| Type of self-attention used in ViT|
| `--fn_places`            | `store_true`|This will freeze the backbone, except for last residual block. Applicable for places finetuning|
| `--pretrained`           |`data-path-to-pretrained-checkpoint`| It will initialise the model's weights from predefined checkpoint|
| `--decoup`               | `store_true`|Freeze Backbone and train only classifier as in decoupled strategy|
| `--deffered`             | `store_true`|Use Deffered Reweighting during training|
| `--sampler`              | `random,upsampling,downsampling`| Use various sampling strategies|

### CIFAR100-LT ρ=100 trainning 
For CIFAR-LT datasets, all models have been trained on 1x V100 GPUs with 
the following parameters:
```
torchrun --nproc_per_node=1 train.py --model se_resnet32 --batch-size 512 --lr 0.2 --lr-warmup-epochs 3 --lr-warmup-method linear --auto-augment cifar10 --epochs 500 --weight-decay 0.001 --mixup-alpha 0.2 --dset_name=cifar100 --output-dir ../experiments/c100_imb100_se_r32_b512_e500_wd1e3_gumbel_iif_se --classif_norm cosine --lr-scheduler cosineannealinglr --imb_factor 0.01 --use_gumbel_se --criterion iif
```


### ImageNet-LT 
For ImageNet-LT dataset, all models have been trained on 4x V100 GPUs. For 200 epoch training use this:
```
torchrun --nproc_per_node=4  train.py --dset_name=imagenet_lt --model se_resnet50 --output-dir ../experiments/se_r50_ilt_ce_mean_wd1e-4_e200_ilt_gumbel_se -b 64 --lr-scheduler cosineannealinglr --reduction mean --lr 0.2 --epochs 200 --mixup 0.2 --auto-augment imagenet --classif_norm cosine --wd 0.0001 --use_gumbel_se --criterion iif
```
For 600 epoch training and stronger results use this:
```
torchrun --nproc_per_node=4 train.py --model se_resnet50 --batch-size 256 --lr 0.5 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment imagenet --epochs 600 --weight-decay 0.0001 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 --val-resize-size 232 --dset_name=imagenet_lt --output-dir ../experiments/se_r50_ilt_ce_mean_wd1e-4_b256_e600_aa_gumbel_se --classif_norm cosine --use_gumbel_se --criterion iif
```
For training ViTs on ImageNet-LT use:
```
torchrun --nproc_per_node=8 train.py --model b_simple_vit --batch-size 32 --classif_norm cosine --lr 0.2 --lr-scheduler cosineannealinglr --lr-warmup-epochs 3 --lr-warmup-method linear --auto-augment imagenet --epochs 200 --weight-decay 1e-5 --mixup-alpha 0.2 --cutmix-alpha 1.0 --dset_name=imagenet_lt --label-smoothing 0.1 --output-dir ../experiments/ilt_vit_b_gumbel --attn gumbel
```

### Places-LT
For Places-LT dataset, first pretrain SE-ResNet152 on full ImageNet. The hyper-parameters are adjusted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/resnext101-32x4d/README.md. 
Use the following command for pretraining:
```
torchrun --nproc_per_node=4 train.py --dset_name=ImageNet --model se_resnet152 --output-dir ../checkpoints/se_r152_inet_e100_aaug_gumbel_se -b 64 --epochs 100 --mixup 0.2 --momentum 0.875 --wd 6.103515625e-05 --lr-scheduler cosineannealinglr --lr-warmup-epochs 3 --lr-warmup-method linear --lr 0.256 --auto-augment imagenet --use_gumbel_se --train-crop-size 192 --val-resize-size 232 
```

Then freeze all parameters except for final Resnet-block and Classifier and train for 40 epochs using:

```
torchrun --nproc_per_node=4 train.py --dset_name places_lt --data-path ../../../datasets/places365_standard/ --model se_resnet152 --epochs 40 -b 64 --lr 0.1 --output-dir ../experiments/se_r152_places_ra_gumbel_se_v2/ --wd 5e-05 --lr-scheduler cosineannealinglr --lr-warmup-epochs 3 --lr-warmup-method linear --pretrained ../experiments/se_r152_inet_e100_aaug_gumbel_se/model_99.pth --mixup 0.2 --auto-augment imagenet --classif_norm lr_cosine --label-smoothing 0.1 --cutmix-alpha 1.0 --use_gumbel_se --criterion iif --fn_places
```

For training ViTs on Places-LT use:
```
torchrun --nproc_per_node=4 train.py --dset_name places_lt --data-path ../../../datasets/places365_standard/ --model s_simple_vit --epochs 15 -b 64 --lr 0.1 --output-dir ../experiments/places_vit/ --wd 5e-05 --lr-scheduler cosineannealinglr --lr-warmup-epochs 3 --lr-warmup-method linear --mixup 0.2 --auto-augment imagenet --classif_norm lr_cosine --label-smoothing 0.1 --cutmix-alpha 1.0 --pretrained ../pretrained/vit_s_cosine/model_199.pth --criterion iif --epochs 15 --attn gumbel
```

### iNaturalist18
For iNaturalist, the model was trained using 4x V100 GPUs with:

```
torchrun --nproc_per_node=4 train.py --model se_resnet50 --data-path ../../../datasets/  --batch-size 256 --lr 0.5 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ra  --epochs 500 --random-erase 0.1 --weight-decay 0.0001 --norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 --val-resize-size 232 --ra-sampler --ra-reps 4 --dset_name=inat18 --output-dir ../experiments/inat_se_r50_softmax_e500_cosine_mixup_wd1e-4_aug_gumbel_se/ --classif_norm cosine --use_gumbel_se
```

For training ViTs on iNaturalist use:
```
torchrun --nproc_per_node=8 train.py --model b_simple_vit --batch-size 128 --lr 0.4 --lr-scheduler cosineannealinglr --lr-warmup-epochs 3 --lr-warmup-method linear --auto-augment ra --epochs 500 --weight-decay 0.00001 --mixup-alpha 0.2 --cutmix-alpha 1.0 --label-smoothing 0.1 --dset_name=inat18 --random-erase 0.1 --ra-sampler --ra-reps 4 --output-dir ../experiments/inat_vit_b_e500_b1024_cos_ra_softmax_gumbel --classif_norm cosine --train-crop-size 192 --val-resize-size 232 --attn gumbel --data-path ../../../datasets/ --apex
```