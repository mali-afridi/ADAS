# based on https://github.com/open-mmlab/mmdetection/blob/v2.28.0/demo/image_demo.py
# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from mmcv.parallel import collate, scatter
import torch
from mmdet.apis import init_detector
from libs.datasets.pipelines import Compose
from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes
from RT_DETR.rtdetr_pytorch.src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_detections(image_path, detections, score_threshold=0.4, save_path=None):
    image = cv2.imread(image_path)
    # image = cv2.resize(image,(640,640))
    for detection in detections:
        labels = detection['labels']
        boxes = detection['boxes']
        scores = detection['scores']

        for label, box, score in zip(labels, boxes, scores):
                if score > score_threshold:
                    xmin, ymin, xmax, ymax = box.tolist()
                    xmin, ymin, xmax, ymax  = int(xmin), int(ymin), int(xmax), int(ymax)

                        # Draw the bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Annotate with the class name and score
                label_text = f"Class {label}, Score {score:.2f}"
                cv2.putText(image, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if save_path:
        cv2.imwrite(save_path, image)  # Save the image
    else:
        cv2.imshow('Object Detections', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default='result.png', help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold'
    )
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    postprocessor = RTDETRPostProcessor(num_top_queries= 300)
    model = init_detector(args.config, args.checkpoint, device='cpu')
    
    model.backbone2 = None
    model.neck = None
    model.bbox_head = None
    weights = torch.load('rtdetr_r50vd.pth')
    weights = weights['ema']['module']
    import pdb; pdb.set_trace()
    model.load_state_dict(weights)
    # test a single image
    model.eval()
    # model.bbox_head.evaluating=True

    
    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessor.iou_types
    img = cv2.imread(args.img)
    ori_shape = img.shape
    data = dict(
        filename=args.img,
        sub_img_name=None,
        img=img,
        gt_points=[],
        id_classes=[],
        id_instances=[],
        img_shape=ori_shape,
        ori_shape=ori_shape,
    )

    cfg = model.cfg
    # model.bbox_head.test_cfg.as_lanes = False
    # device = next(model.parameters()).device  # model device

    test_pipeline = Compose(cfg.data.test.pipeline)

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    data['img_metas'] = data['img_metas'].data[0]
    data['img'] = data['img'].data[0]
    
    img_metas = []
    import pdb; pdb.set_trace()
    with torch.no_grad():
      obj_results=  model(data['img'],img_metas=img_metas,task="object")


    # orig_target_sizes = torch.tensor([ori_shape[0],ori_shape[1]])
    orig_target_sizes = torch.tensor([1640,590])
    det = postprocessor(obj_results, orig_target_sizes)
    # detections = postprocessor(obj_results, orig_target_sizes)
    visualize_detections(args.img, postprocessor(obj_results, orig_target_sizes),save_path='result_rtdetr.png')
    import pdb; pdb.set_trace()
    src, preds = inference_one_image(model, args.img)
    # show the results
    dst = visualize_lanes(src, preds, save_path=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    
