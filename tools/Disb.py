# Adapted from:
# https://github.com/open-mmlab/mmdetection/blob/db256a14bb7018ad36eed104ea0ce178a0d4050c/tools/train.py
# Copyright (c) OpenMMLab. All rights reserved.

import copy

# from mmcv import AdamW, CosineAnnealingLrUpdaterHook
from mmdet.models.builder import build_head,build_neck,build_backbone
from mmcv.utils.config import ConfigDict


from argparse import ArgumentParser


import time
import warnings
import torch.optim
from mmdet.apis import init_detector


from libs.datasets.metrics.culane_metric import interp



import numpy as np
import torch


from RT_DETR.rtdetr_pytorch.src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor



from mmdet import __version__

import cv2

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

from libs.datasets.metrics.culane_metric import interp
import numpy as np

from mmcv.utils import Config
import torch
import cv2
import copy
import time
import os

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
    # parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('data', default='isb.txt',type = str, help='Path of text file to perform inference on')
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


    cfg = Config.fromfile(args.config)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()
    model.bbox_head.evaluating=True

    model = model.to(device='cuda')
    postprocessor = RTDETRPostProcessor(num_top_queries= 300)
    # COCO
    # weight_obj_loss=1.0
    # weight_lane_loss=5.0
    model.bbox_head.evaluating = True
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    file_isb = args.data
    
    file_isb = open(file_isb)
    images_list = file_isb.readlines()
    images_list.sort()
    file_isb.close()
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

    img_metas = [img_metas]
    frame, total, fps = 0, 0, 0
    os.makedirs(f'{args.out_file}',exist_ok=True)
    for i in range(len(images_list)):
            # import pdb; pdb.set_trace()

            inp = cv2.imread('.'+images_list[i][:-1])
            # inp = cv2.imread(test)
            img_metas[0]['filename'] = '.'+images_list[i][:-1] 
            resized_image = cv2.resize(inp, (1640, 590))
            # cv2.imwrite('.'+images_list[i][:-1],resized_image)
            inp = cv2.imread('.'+images_list[i][:-1])
            # inp = cv2.imread('00020.jpg')
            cropped_image = inp[start_y:end_y, start_x:end_x]
            # cv2.imwrite('testcrop.png',cropped_image)
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

            orig_target_sizes = torch.tensor([1640,590]).to(device='cuda')
            results = postprocessor(obj_out, orig_target_sizes)

            src = cv2.imread('.'+images_list[i][:-1])

            
            visualize_detections(src=src,preds=preds, detections=results,save_path=f'{args.out_file}/fyp{i}.png')




    


    


mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}



if __name__ == '__main__':
    args = parse_args()
    main(args)
