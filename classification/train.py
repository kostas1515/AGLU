import datetime
import os
import time
import warnings


import torch
import torch.utils.data
import torchvision
import torchvision.models as models
import torch.distributed as dist

try:
    import presets
    import transforms
    import utils
    from sampler import RASampler
    import resnet_pytorch
    import imbalanced_dataset
    import initialise_model
except ModuleNotFoundError:
    from classification import presets
    from classification import transforms
    from classification import utils
    from classification.sampler import RASampler
    from classification import resnet_pytorch
    from classification import imbalanced_dataset
    from classification import initialise_model
    

from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

from catalyst.data import  BalanceClassSampler,DistributedSamplerWrapper

try:
    from apex import amp
    import apex
except ImportError:
    amp = None
    
    
def record_result(result,args):
    df = pd.DataFrame.from_dict(vars(args))
    df['acc']=result
    file2save=os.path.join(args.output_dir,'results.csv')
    df = df.iloc[1: , :]
    if os.path.exists(file2save):
        df.to_csv(file2save, mode='a', header=False)
    else:
        df.to_csv(file2save)



def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    
    apex=args.apex

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)
            

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        batch_size = image.shape[0]
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            if hasattr(criterion, "iif"):
                output = criterion(output, infer=True)
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()
    
    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg
        

def select_training_param(model):
#     print(model)
    for v in model.parameters():
        v.requires_grad = False
    try:
        torch.nn.init.xavier_uniform_(model.linear.weight)
        model.linear.weight.requires_grad = True
        try:
            model.linear.bias.requires_grad = True
            model.linear.bias.data.fill_(0.01)
        except AttributeError:
            pass
    except AttributeError:
        torch.nn.init.xavier_uniform_(model.fc.weight)
        model.fc.weight.requires_grad = True
        try:
            model.fc.bias.requires_grad = True
            model.fc.bias.data.fill_(0.01)
        except AttributeError:
            pass
    return model


def finetune_places(model):
#     print(model)
    for v in model.parameters():
        v.requires_grad = False
    
    torch.nn.init.xavier_uniform_(model.fc.weight)
    model.fc.weight.requires_grad = True
    try:
        model.fc.bias.requires_grad = True
    except AttributeError:
        pass
    
    for v in model.layer4.parameters():
        v.requires_grad = True
    for name,param in model.layer4.named_parameters():
        if name.endswith('param'):
            init = torch.rand(1).item()
            param.data.fill_(init)
    return model


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = args.ra_magnitude
        augmix_severity = args.augmix_severity
        train_transform = presets.ClassificationPresetTrain(
                    crop_size=train_crop_size,
                    interpolation=interpolation,
                    auto_augment_policy=auto_augment_policy,
                    random_erase_prob=random_erase_prob,
                    ra_magnitude=ra_magnitude,
                    augmix_severity=augmix_severity,
                )
        if args.dset_name == 'ImageNet':
            dataset = torchvision.datasets.ImageFolder(
                traindir,train_transform,
            )
            num_classes = 1000
        elif args.dset_name =="imagenet_lt":
            num_classes = 1000
            train_txt = "../../../datasets/ImageNet-LT/ImageNet_LT_train.txt"
            eval_txt = "../../../datasets/ImageNet-LT/ImageNet_LT_test.txt"
            dataset = imbalanced_dataset.LT_Dataset(args.data_path, train_txt,num_classes, transform=train_transform)
            num_classes = len(dataset.cls_num_list)
        elif args.dset_name =="inat18":
            num_classes = 8142
            # auto_augment_policy = getattr(args, "auto_augment", None)
            train_txt = "../../../datasets/train_val2018/iNaturalist18_train.txt"
            eval_txt = "../../../datasets/train_val2018/iNaturalist18_val.txt"
            dataset = imbalanced_dataset.LT_Dataset(args.data_path, train_txt,num_classes, transform=train_transform)
            num_classes = len(dataset.cls_num_list)
        elif args.dset_name =="places_lt":
            num_classes = 365
            # auto_augment_policy = getattr(args, "auto_augment", None)
            train_txt = "../../../datasets/places365_standard/Places_LT_train.txt"
            eval_txt = "../../../datasets/places365_standard/Places_LT_test.txt"
            dataset = imbalanced_dataset.LT_Dataset(args.data_path, train_txt,num_classes, transform=train_transform)
            num_classes = len(dataset.cls_num_list)
        else:
            dataset, dataset_test = imbalanced_dataset.load_cifar(args)
            num_classes = len(dataset.num_per_cls_dict)
        
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms()
        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
            )
        if args.dset_name == 'ImageNet':
            dataset_test = torchvision.datasets.ImageFolder(
                valdir,
                preprocessing,
            )
        elif args.dset_name.startswith('cifar') is True:
            pass # test dataset already loaded
        else:
            dataset_test = imbalanced_dataset.LT_Dataset_Eval(args.data_path, eval_txt,dataset.class_map, num_classes, transform=preprocessing)
            
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            if args.sampler=='random':
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                train_labels = dataset.targets
                balanced_sampler = BalanceClassSampler(train_labels,mode=args.sampler)
                train_sampler= DistributedSamplerWrapper(balanced_sampler)
        
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        if args.sampler=='random':
            train_sampler = torch.utils.data.RandomSampler(dataset)
        else:
            train_labels = dataset.targets
            train_sampler = BalanceClassSampler(train_labels,mode=args.sampler)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    model = initialise_model.get_model(args,num_classes)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = initialise_model.get_criterion(args,dataset,model)
    
    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("lambda_param", args.bias_weight_decay))
        custom_keys_weight_decay.append(("kappa_param", args.bias_weight_decay))

    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        adam_eps = 1e-5 if args.apex else 1e-8
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay,betas=(0.9,args.adamW2),eps=adam_eps)
    elif opt_name == "lamb":
        optimizer = apex.optimizers.FusedLAMB(parameters, lr=args.lr, weight_decay=args.weight_decay,betas=(0.9,args.adamW2))
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")      
        
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.apex_opt_level
                                      )
        
    if args.decoup:
        model = select_training_param(model)

    if args.fn_places is True:
        model = finetune_places(model)

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler =='multistep':
        main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
                
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])
        elif args.apex:
            amp.load_state_dict(checkpoint["amp"])
            
    if args.load_from:
        checkpoint = torch.load(args.load_from, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"],strict=False)

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.ms_train is True:
            temp_train_crop=torch.randint(8, 20, (1,)).cuda()
            if args.distributed:
                batch_w=[torch.zeros_like(temp_train_crop) for _ in range(dist.get_world_size())]
                dist.all_gather(batch_w,temp_train_crop)
                temp_train_crop=torch.cat(batch_w,axis=0)[0]
                print('Crop is: ',temp_train_crop*16)
                setattr(data_loader.dataset.transform.transforms.transforms[0],'size',(temp_train_crop.item()*16, temp_train_crop.item()*16))
                
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        lr_scheduler.step()
        if args.no_val is False:
            acc = evaluate(model, criterion, data_loader_test, device=device)
            if acc>best_acc:
                best_acc = acc
            if model_ema:
                evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            elif args.apex:
                checkpoint["amp"] = amp.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    print('best acc is:',best_acc)
    if args.record_result:
        if utils.is_main_process():
            record_result(best_acc,args)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="../../../datasets/ILSVRC/Data/CLS-LOC/", type=str, help="dataset path")
    parser.add_argument('--dset_name', default='ImageNet', help='dataset name')
    parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    parser.add_argument("--model", default="resnet32", type=str, help="model name")
    parser.add_argument('--use_gumbel_se', default=False, help='Gumbel activation in excitation phase of SE',action='store_true')
    parser.add_argument('--use_gumbel_cb', default=False, help='Gumbel activation in spatial attention phase of CB',action='store_true')
    parser.add_argument('--classif_norm', default=None,type=str, help='Type of classifier Normalisation {None,norm,cosine')
    parser.add_argument('--criterion', default='ce',type=str, help='Criterion used for classifier {ce,bce,gce')
    parser.add_argument('--ms_train',default=False, action='store_true',
                        help='Use Multi-scale training')
    
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument('--adamW2', type=float, default=0.95)
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for lambda and kappa parameters of all AGCA layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="multistep", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument('--decoup',action="store_true", help='Freeze all layers except classif layer')
    parser.add_argument('--milestones',nargs='+', default=[200,220],type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument("--lr-warmup-epochs", default=3, type=int, help="the number of epochs to warmup (default: 3)")
    parser.add_argument(
        "--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.01, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint, loads model's and optimiser's params and resumes training")
    parser.add_argument("--load_from", default="", type=str, help="path of checkpoint, loads only model weights")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--fn_places",
        help="Finetune last resnet block in places",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--no-val",
        dest="no_val",
        help="Don't run evaluation during training",
        action="store_true",
        default=False
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
    parser.add_argument('--sampler', default='random', type=str, help='sampling, [random,upsampling,downsampling]')
    parser.add_argument('--reduction', default='mean', type=str, help='reduce mini batch')
    parser.add_argument('--iif', default='raw',type=str, help='Type of IIF variant- applicable if classif iif')
    

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--deffered",
        help="Use deferred schedule",
        action="store_true",
    )
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--find-unused-params", action="store_true",default=False, help="store true in DDP for trainning cait")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo or local --> [pytorch, path-to-model]",
        default=None, type=str,
    )
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O2', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )
    parser.add_argument('--record-result', dest="record_result",
        help="Record result in csv format",
        action="store_true")
    
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)