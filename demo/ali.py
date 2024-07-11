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
import cv2, os
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
    parser.add_argument('--out-file', default='result_dl', help='Path to output file')
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
    os.makedirs(f'{args.out_file}',exist_ok=True)
    ops =[
        # {'type': 'RandomPhotometricDistort', 'p': 0.5},
        #   {'type': 'RandomZoomOut', 'fill': 0}, 
        #   {'type': 'RandomIoUCrop', 'p': 0.8}, 
        #    {'type': 'SanitizeBoundingBox', 'min_size': 1},
        #     {'type': 'RandomHorizontalFlip'}, 
            {'type': 'Resize', 'size': [640, 640]},
            {'type': 'ToImageTensor'}, 
            {'type': 'ConvertDtype'}, 
            {'type': 'SanitizeBoundingBox', 'min_size': 1}, 
            {'type': 'ConvertBox', 'out_fmt': 'cxcywh', 'normalize': True}]

    culaneyolo_transforms = Compose(ops)
    # build the model from a config file and a checkpoint file
    postprocessor = RTDETRPostProcessor(num_top_queries= 300)
    cfg = Config.fromfile(args.config)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()
    model.bbox_head.evaluating=True
    # test a single image
    # model.eval()
    culaneyolo_test = CulaneYolo( 
    data_root = cfg.data.val['data_root'],
    # img_folder=cfg.data.train['data_root'],
    data_list = cfg.data.val['data_list'],
    pipeline=cfg.data.val['pipeline'],
    # pipeline=[],
    ann_file = 'dataset2/culaneyolo/docker_format/custom_val_docker.json',
    transforms=culaneyolo_transforms,
    # diff_file=cfg.data.train['diff_file'],
    test_mode=False,
    return_masks=False,
    remap_mscoco_category=True,
    culaneyolotest=True)

    data_loader = DataLoader(
        culaneyolo_test,batch_size=1,shuffle = False,num_workers=4, collate_fn=lambda x: x
    )

    frame = 0
    total = 0

    with torch.no_grad():
        for batch in (data_loader):
            # batch.to(device='cuda')
            targets= []
            #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            img_metas = []
                    #images in batch=4
            for i in range(len(batch)):
                # print(len(batch))
                if (i):
                    img = torch.cat((img,batch[i][0].unsqueeze(0)))
                    img2 = torch.cat((img2,batch[i][3].unsqueeze(0)))

                else:
                    # print("first here")
                    img = (batch[0][0]).unsqueeze(0)
                    img2 = (batch[0][3]).unsqueeze(0)

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
            img2 = img2.to(device='cuda')
            for i2 in range(len(targets)):
                    if (len(targets[i2]['boxes'].shape)==1):
                        targets[i2]['labels']= targets[i2]['labels'].unsqueeze(0)
                    targets[i2]['size'] = torch.tensor([640,640]).to(device='cuda')

            # model = model.to(device='cuda')

            frame +=1

            start = time.time()
            #ye output output sirf
            outputs = model(img=img2, img_metas= img_metas,task="object",targets=targets)
            #ye lanes
           
            results2= model.forward_test(img=img,img_metas=img_metas)
            end = time.time()

            diff = end - start
            
            total = total + diff

            if frame % 100 == 0:
                print("frames = ", frame)
                print(f"Total Time for {frame} Frames: ", total)
                
                fps = frame / total
                print(f"FPS for {frame} Frames: ", fps)
                print("   ")




            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
            results = postprocessor(outputs, orig_target_sizes)


            lanes = results2[0]['result']['lanes']
            ori_shape = (590,1640)
            preds = get_prediction(lanes, ori_shape[0], ori_shape[1])
            src = img_metas[0]['filename']
            src = cv2.imread(src)
            

            visualize_detections(src=src,preds=preds, detections=results,save_path=f'{args.out_file}/{frame}.jpg')




    dst = visualize_lanes(src, preds, save_path=f'{args.out_file}/{frame}.jpg')

if __name__ == '__main__':
    args = parse_args()
    main(args)