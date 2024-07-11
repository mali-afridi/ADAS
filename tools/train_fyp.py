# Adapted from:
# https://github.com/open-mmlab/mmdetection/blob/db256a14bb7018ad36eed104ea0ce178a0d4050c/tools/train.py
# Copyright (c) OpenMMLab. All rights reserved.
import json
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import torch.optim
import random
from collections import OrderedDict
from torchvision.transforms import ToTensor
from RT_DETR.rtdetr_pytorch.src.misc import (MetricLogger, SmoothedValue, reduce_dict)
from RT_DETR.rtdetr_pytorch.src.data import CocoEvaluator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import DistributedSampler

from RT_DETR.rtdetr_pytorch.src.zoo.rtdetr.rtdetr_criterion import SetCriterion
from RT_DETR.rtdetr_pytorch.src.data.transforms import ConvertBox, Compose
from RT_DETR.rtdetr_pytorch.src.zoo.rtdetr.matcher import HungarianMatcher
from RT_DETR.rtdetr_pytorch.src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
from RT_DETR.rtdetr_pytorch.src.data import get_coco_api_from_dataset

from libs.datasets.coco_dataset import CocoDetection
from libs.datasets.culaneyolo import CulaneYolo
from libs.datasets.pipelines import CollectCLRNet
# from RT_DETR.rtdetr_pytorch.src.data.transforms import ConvertBox
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash, mkdir_or_exist
from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataloader, build_dataset
# from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_device, get_root_logger

from mmdet.utils import get_root_logger
import torchvision.transforms.v2 as T
# from mmdet.datasets.transforms import RandomPhotometricDistort, RandomZoomOut, RandomIoUCrop, SanitizeBoundingBox, ConvertBox
# from mmdet.datasets.utils import default
# suppress mmcv v2.0.0 announcement
warnings.filterwarnings("ignore", module="mmcv")


def is_dist_available_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


RandomPhotometricDistort = (T.RandomPhotometricDistort)
RandomZoomOut = (T.RandomZoomOut)
RandomIoUCrop = (T.RandomIoUCrop)
RandomHorizontalFlip = (T.RandomHorizontalFlip)
Resize = (T.Resize)
ToImageTensor = (T.ToImageTensor)
ConvertDtype = (T.ConvertDtype)
SanitizeBoundingBox = (T.SanitizeBoundingBox)
RandomCrop = (T.RandomCrop)
Normalize = (T.Normalize)

ops =[#{'type': 'RandomPhotometricDistort', 'p': 0.5},
      # {'type': 'RandomZoomOut', 'fill': 0}, 
      # {'type': 'RandomIoUCrop', 'p': 0.8}, 
       #{'type': 'SanitizeBoundingBox', 'min_size': 1},
       #  {'type': 'RandomHorizontalFlip'}, 
         {'type': 'Resize', 'size': [640, 640]},
           {'type': 'ToImageTensor'}, 
           {'type': 'ConvertDtype'}, 
           {'type': 'SanitizeBoundingBox', 'min_size': 1}, 
           {'type': 'ConvertBox', 'out_fmt': 'cxcywh', 'normalize': True}]

culaneyolo_transforms = Compose(ops)
def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="ids of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('checkpoint', help='Checkpoint file')
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both "
            "specified, --options is deprecated in favor of --cfg-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --cfg-options")
        args.cfg_options = args.options

    return args

# import wandb
# wandb.init(
# # set the wandb project where this run will be logged
# project="Weighted CLRerNet Transformer",

# )

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
        if len(args.gpu_ids) == 1:
            torch.cuda.set_device(args.gpu_ids[0])
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text
    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if args.seed is not None:
        logger.info(
            f"Set random seed to {args.seed}, " f"deterministic: {args.deterministic}"
        )
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta["seed"] = args.seed
    meta["exp_name"] = osp.basename(args.config)
    # import pdb; pdb.set_trace()

    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    
    #similar.sum() -> useless
    model.init_weights()
    #temporarily disable trasnforms
    # cfg.data.train['pipeline'] = []
    # cfg.train.train['transforms'] = culaneyolo_transform
    # culaneyolo_transform = 
    culaneyolo = CulaneYolo( 
        data_root = cfg.data.train['data_root'],
        # img_folder=cfg.data.train['data_root'],
        data_list = cfg.data.train['data_list'],
        pipeline=cfg.data.train['pipeline'],
        # pipeline=[],
        ann_file = 'data/CULane Complete/culaneyolo/frame_diff/frame_train_docker.json',
        transforms=culaneyolo_transforms,
        diff_file=cfg.data.train['diff_file'],
        test_mode=False,
        return_masks=False,
        remap_mscoco_category=True,
        culaneyolotest=False)
    
    culaneyolo_test = CulaneYolo( 
        data_root = cfg.data.val['data_root'],
        # img_folder=cfg.data.train['data_root'],
        data_list = cfg.data.val['data_list'],
        pipeline=cfg.data.val['pipeline'],
        # pipeline=[],
        ann_file = 'data/CULane Complete/culaneyolo/docker_format/custom_val_docker.json',
        transforms=culaneyolo_transforms,
        # diff_file=cfg.data.train['diff_file'],
        test_mode=False,
        return_masks=False,
        remap_mscoco_category=True,
        culaneyolotest=True)
   
    # import pdb; pdb.set_trace()
    # culane_dataset = [build_dataset(cfg.data.train)]
    batch_size=10
    data_loader = DataLoader(
        culaneyolo,batch_size=batch_size,shuffle = True,num_workers=4, collate_fn=lambda x: x
    )
    val_dataloader = DataLoader(
        culaneyolo_test,batch_size=batch_size,shuffle = False,num_workers=4, collate_fn=lambda x: x
    )
    matcher = HungarianMatcher(
        weight_dict= {'cost_class': 2, 'cost_bbox': 2, 'cost_giou': 10},
        use_focal_loss= False,
        alpha= 0.25,
        gamma= 2.0
    ) 
    criterion = SetCriterion(
        weight_dict= {'loss_vfl': 1, 'loss_bbox':2, 'loss_giou': 10},
        losses= ['vfl', 'boxes'],
        alpha= 0.25,
        gamma= 2.0,
        matcher = matcher
    )
    # model.train()
    # import pdb; pdb.set_trace()
    initial_lr = 0.000000

    optimizer = torch.optim.AdamW(params = model.parameters(), lr=initial_lr,weight_decay=0.0001)
    # initial_lr = 0.000000
    final_lr = 0.000100
    lr_increment = 0.000005
    current_lr = initial_lr

    epochs = 70
    model = init_detector(args.config, args.checkpoint, device='cpu')
    # test a single image
    model.eval()
    model.bbox_head.evaluating=True

    model = model.to(device='cuda')

    postprocessor = RTDETRPostProcessor(num_top_queries= 300)
    # COCO
    weight_obj_loss=1.0
    weight_lane_loss=5.0
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    base_ds = get_coco_api_from_dataset(val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
    best_stat = {'epoch': -1, }

    for epoch in range(epochs):

        # model.train()
        # criterion.train()
        # model.bbox_head.evaluating=False
        iteration,mean_total_loss,mean_obj_loss,mean_loss_lane=0,0,0,0
        if(weight_lane_loss<=1.0):
            weight_lane_loss-=0.1
        if(weight_obj_loss!=1.0):
            weight_obj_loss-=0.1

        # test_stats, coco_evaluator = evaluate(
        #     model, criterion, postprocessor, val_dataloader, base_ds
        # )
       
        # for batch in (data_loader):

        #     # batch.to(device='cuda')
        #     targets= []
        #     #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #     img_metas = []
        #     optimizer.zero_grad()
        #     # import pdb; pdb.set_trace()
        #             #images in batch=4
        #     for i in range(len(batch)):
        #         # print(model.bbox_head.evaluating)
        #         if (i):
        #             img = torch.cat((img,batch[i][0].unsqueeze(0)))
        #         else:
        #             # print("first here")
        #             img = (batch[0][0]).unsqueeze(0)
        #         target=batch[i][2]

        #         if (len(target['boxes'].shape)>2):
        #                 target['boxes'] = target['boxes'].squeeze(0)
        #         target['boxes']=target['boxes'].to(device='cuda')
        #         target['labels'] = target['labels'].squeeze(0).to(device='cuda')
        #         target['image_id'] = target['image_id'].squeeze(0).to(device='cuda')
        #         target['area'] = target['area'].squeeze(0).to(device='cuda')
        #         target['iscrowd'] = target['iscrowd'].squeeze(0).to(device='cuda')
        #         target['orig_size'] = target['orig_size'].squeeze(0).to(device='cuda')
        #         target['size'] = target['size'].squeeze(0).to(device='cuda')
        #         # import pdb; pdb.set_trace()
        #         targets.append(target)
        #         img_metas.append(batch[i][1])

        #         img_metas[i]['lanes'] = (img_metas[i]['lanes']).to(device='cuda')
        #             # Update the learning rate of the optimizer
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = current_lr    
        #     img = img.to(device='cuda')
        #     for i2 in range(len(targets)):
        #             if (len(targets[i2]['boxes'].shape)==1):
        #                 targets[i2]['labels']= targets[i2]['labels'].unsqueeze(0)
        #             targets[i2]['size'] = torch.tensor([640,640]).to(device='cuda')


        #     loss_lane, obj_out = model(img=img, img_metas= img_metas,targets=targets)
            
        #     for i1 in range(len(targets)):
        #         if (targets[i1]['labels'].shape[0]==1):
        #             targets[i1]['boxes']=targets[i1]['boxes'].unsqueeze(0)
        #     obj_loss = criterion(obj_out, targets)
        #     # for key, value in obj_loss.items():
        #     #     print(f"{key}: {value.item():.2f}")
            
        #     # import pdb; pdb.set_trace()
                
        #     loss = sum(obj_loss.values())
        #     loss2 = sum(loss_lane.values())
        #     total_loss = (weight_obj_loss*(loss))+(weight_lane_loss*(loss2)) #loss weight
        #     total_loss.backward()
        #     optimizer.step()

        #     # import pdb; pdb.set_trace()
        #     mean_obj_loss+=loss.item()
        #     mean_loss_lane+=loss2.item()
        #     mean_total_loss+=(loss.item()+loss2.item())

        #     if((iteration%100) ==0):
        #         formatted_values = " | ".join([f"{key}: {value.item():.2f}" for key, value in obj_loss.items()])
        #         print(formatted_values)
        #         print(f"Epoch: {epoch} Iteration:{iteration}/{len(data_loader)} and learning rate: {current_lr}")
        #         print(f"Lane loss weight:{weight_lane_loss} and object loss weight: {weight_obj_loss}")
        #         print(f"Total Obj Loss: {'{:.3f}'.format(loss.item())}")
        #         print(f"Total Lane Loss: {'{:.3f}'.format(loss2.item())}")
        #         wandb.log({"Iter obj_loss": loss.item(), 'Iter lane_loss:':loss2.item()})
        #         current_lr += lr_increment
        #         current_lr = min(current_lr, final_lr)

        #     iteration+=1
            
            # import pdb; pdb.set_trace()

            # print(f"Epoch: ")
        # wandb.log({"total mean loss": mean_total_loss/len(data_loader), "Mean obj_loss": mean_obj_loss/len(data_loader), 'Mean lane_loss:':mean_loss_lane/len(data_loader)})
        # if (((epoch+1)%10) ==0):
        # torch.save(model.state_dict(),f"CLRerNet_Transformer{epoch}.pth")
        
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessor, val_dataloader, base_ds
        )
            # # TODO 
        for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
        print('best_stat: ', best_stat)
        log_stats = {
                    #  {f'train_{k}': v for k, v in train_stats.items()},
        **{f'test_{k}': v for k, v in test_stats.items()},
        'epoch': epoch,
        'n_parameters': n_parameters}

        with open("logs.txt","a") as f:
            f.write(json.dumps(log_stats) + "\n")

        print(f"Epoch {epoch+1}:**{i}/{len(data_loader)}** \n Obj loss:{'{:.3f}'.format(mean_obj_loss/len(data_loader))} and lane loss: {'{:.3f}'.format(mean_loss_lane/len(data_loader))} Total loss: {'{:.3f}'.format(mean_total_loss/len(data_loader))}")
            # i+=1

    import pdb; pdb.set_trace()
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        dataset.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7], CLASSES=dataset[0].CLASSES
        )
    cfg.device = get_device()

    # add an attribute for visualization convenience
    model.CLASSES = dataset[0].CLASSES
    

    logger = get_root_logger(cfg.log_level)
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # data_loaders = [

    
 

@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds):
    model.eval()
    model.bbox_head.evaluating=True
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    # import pdb; pdb.set_trace()
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    iter=1
    print("Evaluating on Val...")
    with torch.no_grad():
        for batch in (data_loader):
            # batch.to(device='cuda')
            targets= []
            #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            img_metas = []
            # import pdb; pdb.set_trace()
                    #images in batch=4
            for i in range(len(batch)):
                # print(len(batch))
                if (i):
                    img = torch.cat((img,batch[i][0].unsqueeze(0)))
                else:
                    # print("first here")
                    img = (batch[0][0]).unsqueeze(0)
                target=batch[i][2]

                if (len(target['boxes'].shape)>2):
                        target['boxes'] = target['boxes'].squeeze(0)
                target['boxes']=target['boxes'].to(device='cuda')
                target['labels'] = target['labels'].squeeze(0).to(device='cuda')
                target['image_id'] = target['image_id'].squeeze(0).to(device='cuda')
                target['area'] = target['area'].squeeze(0).to(device='cuda')
                target['iscrowd'] = target['iscrowd'].squeeze(0).to(device='cuda')
                target['orig_size'] = target['orig_size'].squeeze(0).to(device='cuda')
                target['size'] = target['size'].squeeze(0).to(device='cuda')

                targets.append(target)
                img_metas.append(batch[i][1])

                # img_metas[i]['lanes'] = (img_metas[i]['lanes']).to(device='cuda')

            img = img.to(device='cuda')
            for i2 in range(len(targets)):
                    if (len(targets[i2]['boxes'].shape)==1):
                        targets[i2]['labels']= targets[i2]['labels'].unsqueeze(0)
                    targets[i2]['size'] = torch.tensor([640,640]).to(device='cuda')

            # import pdb; pdb.set_trace()

            _, outputs = model(img=img, img_metas= img_metas,targets=targets)

            for i1 in range(len(targets)):
                if (targets[i1]['labels'].shape[0]==1):
                    targets[i1]['boxes']=targets[i1]['boxes'].unsqueeze(0)
            obj_loss = criterion(outputs, targets)
                
            loss = sum(obj_loss.values())
            # wandb.log({"Val obj_loss": loss.item()})

            print(f"Iter{iter}\{len(data_loader)}: Eval  obj loss:{'{:.3f}'.format(loss.item())}")
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
            results = postprocessors(outputs, orig_target_sizes)
            # if (iter==224):
            #     import pdb; pdb.set_trace()
            res = {int(str(target['leftover'].item())+str(target['image_id'].item())): output for target, output in zip(targets, results)}
            if coco_evaluator is not None:
                coco_evaluator.update(res)
            iter+=1
            # print(f"{iter}/{len(data_loader)}")
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator

if __name__ == "__main__":
    main()
