# Adapted from:
# https://github.com/open-mmlab/mmdetection/blob/db256a14bb7018ad36eed104ea0ce178a0d4050c/tools/train.py
# Copyright (c) OpenMMLab. All rights reserved.
import json
import argparse
import copy
import os
# from mmcv import AdamW, CosineAnnealingLrUpdaterHook
from mmdet.models.builder import build_head,build_neck,build_backbone
from mmcv.utils.config import ConfigDict

import torchvision
from argparse import ArgumentParser

import os.path as osp
import time
import warnings
import torch.optim
from mmdet.apis import init_detector

import random
from collections import OrderedDict
from torchvision.transforms import ToTensor
# from libs.models.backbones.presnet import PResNet
from libs.models.layers.attentions import ROIGather
from libs.models.losses.seg_loss import CLRNetSegLoss
from libs.datasets.metrics.culane_metric import interp

from libs.models.losses.iou_loss import LaneIoULoss
from libs.models.losses.focal_loss import KorniaFocalLoss
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss
from libs.models.dense_heads.seg_decoder import SegDecoder
from libs.core.anchor.anchor_generator import CLRerNetAnchorGenerator
from libs.models.necks import CLRerNetFPN
from libs.models.dense_heads import CLRerHead
from RT_DETR.rtdetr_pytorch.src.misc import (MetricLogger, SmoothedValue, reduce_dict)
from RT_DETR.rtdetr_pytorch.src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from RT_DETR.rtdetr_pytorch.src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from libs.models.backbones.presnet import PResNet

from RT_DETR.rtdetr_pytorch.src.data import CocoEvaluator
from torch.optim.lr_scheduler import CosineAnnealingLR

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
from libs.models.backbones.dla import DLANet
from libs.datasets.coco_dataset import CocoDetection
from libs.datasets.culaneyolo import CulaneYolo
from libs.datasets.pipelines import CollectCLRNet
from torchvision import datapoints
# from RT_DETR.rtdetr_pytorch.src.data.transforms import ConvertBox
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash, mkdir_or_exist
from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataloader, build_dataset
# from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_device, get_root_logger
import cv2
from mmdet.utils import get_root_logger
import torchvision.transforms.v2 as T
# from mmdet.datasets.transforms import RandomPhotometricDistort, RandomZoomOut, RandomIoUCrop, SanitizeBoundingBox, ConvertBox
# from mmdet.datasets.utils import default
# suppress mmcv v2.0.0 announcement
warnings.filterwarnings("ignore", module="mmcv")
# based on https://github.com/open-mmlab/mmdetection/blob/v2.28.0/demo/image_demo.py
# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet.apis import init_detector
from RT_DETR.rtdetr_pytorch.src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
from torch.utils.data import DataLoader
from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes
from libs.datasets.metrics.culane_metric import interp
import numpy as np
from libs.datasets.culaneyolo import CulaneYolo
from RT_DETR.rtdetr_pytorch.src.data.transforms import Compose
from mmcv.utils import Config
import torch
import cv2
import copy
import time

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 0, 255)
PRED_MISS_COLOR = (0, 0, 255)


def draw_lane(lane, img=None, img_shape=None, width=30, color=(255, 255, 255)):
    """
    Overlay lanes on the image.
    Args:
        lane (np.ndarray): N (x, y) coordinates from a single lane, shape (N, 2).
        img (np.ndarray): Source image.
        img_shape (tuple): Blank image shape used when img is None.
        width (int): Lane thickness.
        color (tuple): Lane color in BGR.
    Returns:
        img (np.ndarray): Output image.
    """
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color, thickness=width)
    return img

def visualize_detections(
        # src,
        # preds, 
        src,
        preds,
        detections, 
        score_threshold=0.6,
        iou_thr=0.3,
        annos=list(),
        concat_src=False,
        save_path=None,
        pred_ious=None,):

    # image = cv2.imread(image_path)
    dst = copy.deepcopy(src)
    for anno in annos:
        dst = draw_lane(anno, dst, dst.shape, width=4, color=GT_COLOR)
    if pred_ious == None:
        hits = [True for i in range(len(preds))]
    else:
        hits = [iou > iou_thr for iou in pred_ious]
    for pred, hit in zip(preds, hits):
        color = PRED_HIT_COLOR if hit else PRED_MISS_COLOR
        dst = draw_lane(pred, dst, dst.shape, width=4, color=color)
    if concat_src:
        dst = np.concatenate((src, dst), axis=0)
    # if save_path:
    #     cv2.imwrite(save_path, dst)
    # return dst

    # image = cv2.resize(image,(640,640))
    for detection in detections:
        labels = detection['labels']
        boxes = detection['boxes']
        scores = detection['scores']
        # import pdb; pdb.set_trace()
        for label, box, score in zip(labels, boxes, scores):
                
                if score > score_threshold:
                     xmin,ymin, xmax, ymax = box.tolist()
                     xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                     if ((xmax>1630) and (ymax>580)):
                         if (xmin<1640/2):
                             continue
                        # Draw the bounding box
                
                     cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), (0, 255, 255), 5)

                # Annotate with the class name and score
                     label_text = f"Class {label}, Score {score:.2f}"
                     cv2.putText(dst, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if save_path:
        cv2.imwrite(save_path, dst)  # Save the image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default='result.png', help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold'
    )
    args = parser.parse_args()
    return args

def get_prediction(lanes, ori_h, ori_w):
    preds = []
    for lane in lanes:
        # lane = np.array(lane)
        # import pdb; pdb.set_trace()
        # lane = lane.cpu().numpy()
        lane = lane.points
        xs = lane[:, 0]
        ys = lane[:, 1]
        valid_mask = (xs >= 0) & (xs < 1)
        xs = xs * ori_w
        lane_xs = xs[valid_mask]
        lane_ys = ys[valid_mask] * ori_h
        lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
        pred = [(x, y) for x, y in zip(lane_xs, lane_ys)]
        interp_pred = interp(pred, n=5)
        preds.append(interp_pred)
    return preds

def main(args):

    # args = parse_args()
    cfg = Config.fromfile(args.config)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()
    model.bbox_head.evaluating=True
# from mmcv import AdamW, CosineAnnealingLrUpd
    # scheduler = CosineAnnealing
    
    # weights2 = torch.load('CLRerNet_Transformer(obj)0.pth')
    # model.load_state_dict(weights2)
    # print("loaded new weights")
    # resume = 1
    model = model.to(device='cuda')
    postprocessor = RTDETRPostProcessor(num_top_queries= 300)
    # COCO
    # weight_obj_loss=1.0
    # weight_lane_loss=5.0
    model.bbox_head.evaluating = True
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # import pdb; pdb.set_trace()
    # file_isb = 'isb.txt'
    # file_isb = open(file_isb)
    # images_list = file_isb.readlines()
    # images_list.sort()
    # # import pdb; pdb.set_trace()
    # file_isb.close()
    # Define the cropping dimensions
    start_x = 0
    end_x = 1640
    start_y = 270
    end_y = 590
    postprocessor = RTDETRPostProcessor(num_top_queries= 300)
# Crop the image
    model.eval()
    test = 'google.jpg'
    inp = cv2.imread(test)
    img_metas = {
    'filename': 'data/CULane Complete/CULane/driver_23_30frame/05171102_0766.MP4/00020.jpg',
    'sub_img_name': 'driver_23_30frame/05171102_0766.MP4/00020.jpg',
    'ori_shape': (590, 1640, 3),
    'img_shape': (320, 800, 3),
    'img_norm_cfg': {'mean': [0.0, 0.0, 0.0], 'std': [255.0, 255.0, 255.0], 'to_rgb': False},
    'batch_input_shape': (320, 800)
    }
    # [{'filename': 'dataset2/culane/driver_23_30frame/05171102_0766.MP4/00020.jpg', 
    # 'sub_img_name': 'driver_23_30frame/05171102_0766.MP4/00020.jpg', 
    # 'ori_shape': (590, 1640, 3), 
    # 'img_shape': (320, 800, 3), 
    # 'img_norm_cfg': {'mean': array([0., 0., 0.], dtype=float32), 'std': array([255., 255., 255.], dtype=float32), 'to_rgb': False}, 
    # 'batch_input_shape': (320, 800)}]
    
    # crop_bbox = [0, 270, 1640, 590]
    img_metas = [img_metas]
    frame, total, fps = 0, 0, 0
    cap = cv2.VideoCapture(1)
    import pdb; pdb.set_trace()
    while True:
            # import pdb; pdb.set_trace()
            _, frames = cap.read()
            inp = frames
            # inp = cv2.imread(test)
            img_metas[0]['filename'] = 'frame.png' 
            resized_image = cv2.resize(inp, (1640, 590))
            cv2.imwrite('frame.png',resized_image)
            inp = cv2.imread('frame.png')
            # inp = cv2.imread('00020.jpg')
            cropped_image = inp[start_y:end_y, start_x:end_x]
            cv2.imwrite('testcrop.png',cropped_image)
            # import pdb; pdb.set_trace()
            resized_image2 = cv2.resize(cropped_image, (800, 320))
            resized_image = cv2.resize(inp, (640, 640))
            resized_image_float = resized_image.astype(np.float32)
            resized_image2_float = resized_image2.astype(np.float32)
            # Normalize the image
            mean = np.array([0.0, 0.0, 0.0])
            std = np.array([255.0, 255.0, 255.0])
            #obj image
            normalized_image = (resized_image_float - mean) / std
            normalized_image = normalized_image.transpose(2,0,1)
            normalized_image = torch.tensor(normalized_image,dtype= torch.float32).to(device= 'cuda')
            normalized_image = normalized_image.unsqueeze(0)


            normalized_image2 = (resized_image2_float - mean) / std          
            normalized_image2 = normalized_image2.transpose(2,0,1)
            normalized_image2 = torch.tensor(normalized_image2,dtype= torch.float32).to(device= 'cuda')
            normalized_image2 = normalized_image2.unsqueeze(0)
            # import pdb; pdb.set_trace()

            frame +=1
            start = time.time()
            obj_out,lane_out = model(img = normalized_image,img2 = normalized_image2,img_metas=img_metas,task='both')

            end = time.time()
            diff = end - start  
            total = total + diff

            if frame % 10 == 0:
                print(f"Total Time for {frame} Frames: {total:.2f}s")
                
                fps = frame / total
                fps = "{:.2f}".format(fps)

            print("   ")
            print(f"FPS for {frame} Frames: ", fps)

            lanes = lane_out[0]['result']['lanes']
            ori_shape = (590,1640)
            preds = get_prediction(lanes, ori_shape[0], ori_shape[1])
            # obj_out = model.obj_test(normalized_image,task='object')
            orig_target_sizes = torch.tensor([1640,590]).to(device='cuda')
            results = postprocessor(obj_out, orig_target_sizes)

            
            # results[0]["boxes"] = bbox_pred_ali.squeeze(0)
            # import pdb; pdb.set_trace()
            # time.sleep(0.2)
            src = cv2.imread('frame.png')

            # visualize_detections('.'+images_list[i][:-1], results,save_path='ali.png')
            visualize_detections(src=src,preds=preds, detections=results,save_path='ali.png')

            # visualize_detections('.'+images_list[i][:-1], results,save_path='ali.png')
            # import pdb; pdb.set_trace()



    


    



if __name__ == '__main__':
    args = parse_args()
    main(args)
