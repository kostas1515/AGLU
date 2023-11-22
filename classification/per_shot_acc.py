import torch

import resnet_pytorch
import imbalanced_dataset 
import numpy as np
import utils
import argparse
from itertools import chain
import os
try:
    from apex import amp
except ImportError:
    amp = None
import train
import initialise_model

def main(args):
    dset_name = args.dset_name
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, _, test_sampler = train.load_data(train_dir, val_dir, args)

    num_classes = len(dataset.cls_num_list)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)
    
    model = initialise_model.get_model(args,num_classes)
    criterion = initialise_model.get_criterion(args,dataset)
    checkpoint = torch.load(args.load_from, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    model.to('cuda')
    if args.apex:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.apex_opt_level
                                          )

    avg_acc,preds,targets=evaluate(model.cuda(), data_loader_test, device='cuda',criterion=criterion)

    print(f'Avg Acc is: {avg_acc}')

    f,c,r = shot_acc(np.array(preds),np.array(targets),dataset.targets)

    print(f'Many shot Acc is: {f}, median shot Acc is: {c}, low shot Acc is: {r}')

    return 0



def shot_acc (preds, labels, train_targets, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    
    if isinstance(train_targets, np.ndarray):
        training_labels = np.array(train_targets).astype(int)
    else:
        training_labels = np.array(train_targets).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))    
 
    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)



def evaluate(model, data_loader, device, print_freq=10,criterion=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    predictions=[]
    targets=[]
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            if hasattr(criterion, 'iif'):
                output=criterion(output,infer=True)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                
            predictions=list(chain(predictions,output.argmax(axis=1).tolist()))
            targets=list(chain(targets,target.tolist()))
            
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
#             metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    return metric_logger.acc1.global_avg,predictions,targets

def get_args_parser():
    parser = argparse.ArgumentParser(description='Parse arguments for per shot acc.')
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--dset_name', default='imagenet_lt',type=str, help='Dataset Name imagenet_lt|places')
    parser.add_argument(
        '--data-path', default='../../../datasets/ILSVRC/Data/CLS-LOC/', help='dataset')
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
    parser.add_argument('--sampler', default='random', type=str, help='sampling, [random,upsampling,downsampling]')
    parser.add_argument('--iif', default='raw',type=str, help='Type of IIF variant')
    parser.add_argument('--classif', default='ce',type=str, help='Type of classification')
    parser.add_argument('--classif_norm', default=None,type=str, help='Type of classifier Normalisation {None,norm,cosine')
    parser.add_argument('--load_from', default='', help='load wweights only from checkpoint')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 16)')
    parser.add_argument('-b', '--batch-size', default=256, type=int)
    parser.add_argument('--model', default='resnet50', help='model, E.g.(se_resnext50_32x4d)')
    parser.add_argument('--use_gumbel_se', default=False, help='Gumbel activation in excitation phase of SE',action='store_true')
    parser.add_argument('--use_gumbel_cb', default=False, help='Gumbel activation in spatial gate phase of CB',action='store_true')
    
    parser.add_argument('--criterion', default='ce',type=str, help='Criterion used for classifier {ce,bce,gce')
    
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
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
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument('--apex-opt-level', default='O2', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--deffered",
        help="Use deferred schedule",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo or local --> [pytorch, path-to-model]",
        default=None, type=str,
    )
    parser.add_argument("--weights", default=None, type=str, help="Dummy input, do not change")
    parser.add_argument('--reduction', default='mean', type=str, help='reduce mini batch')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Dummy input, do not change",
        action="store_true",
    )


    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
    print('end of program')