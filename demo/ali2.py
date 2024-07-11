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

def main(args):
    
    device = args.device
    # cfg = Config.fromfile(args.config)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # import pdb; pdb.set_trace()
    model.eval()

    dummy_input1 = torch.randn(1, 3, 640, 640).to(device)
    # dummy_input2 = torch.randn(10, 3, 320, 800).to(device)
    img_metas = {
    'filename': 'dataset2/culane/driver_23_30frame/05171102_0766.MP4/00020.jpg',
    'sub_img_name': 'driver_23_30frame/05171102_0766.MP4/00020.jpg',
    'ori_shape': (590, 1640, 3),
    'img_shape': (320, 800, 3),
    'img_norm_cfg': {
        'mean': [0., 0., 0.],
        'std': [255., 255., 255.],
        'to_rgb': False
    },
    'batch_input_shape': (320, 800)
    }

    task = "object"
    targets = None

    img_metas = [img_metas]

    import pdb; pdb.set_trace()
    input_names=['actual_input']
    # input_names = []
    output_names=['results']

    torch.onnx.export(model,
                # args=(dummy_input1, dummy_input2),
                (dummy_input1, img_metas,task),
                "transCLR.onnx",
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                export_params=True,
                opset_version=16
                )

    # test a single image
    # model.eval()

if __name__ == '__main__':
    args = parse_args()
    main(args)