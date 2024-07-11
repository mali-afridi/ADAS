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
import streamlit as st
from PIL import Image

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 0, 255)
PRED_MISS_COLOR = (0, 0, 255)

st.set_page_config(layout="wide")
st.title("Transformer Based ADAS")
st.write("---")

sidebar_options = ["Inference"]
st.sidebar.success("Advanced Driving Assitance System")
st.sidebar.write('---')
st.sidebar.title("Options")
box = st.sidebar.selectbox(" ", sidebar_options)

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
        src,
        preds,
        detections, 
        score_threshold=0.6,
        iou_thr=0.3,
        annos=list(),
        FPS=0,
        concat_src=False,
        save_path=None,
        pred_ious=None,):

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

    num_lanes = len(preds) - 1

    if num_lanes < 0:
        num_lanes = 0

    cv2.putText(dst, f"Number of Lanes: {str(num_lanes)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(dst, f"FPS: {str(FPS)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # Objects Detected
    coco = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic_light': 9, 'fire_hydrant': 10, 'stop_sign': 11, 'parking_meter': 12}
    objects = {'person': 0, 'bicycle': 0, 'car': 0, 'motorcycle': 0, 'airplane': 0, 'bus': 0, 'train': 0, 'truck': 0, 'boat': 0, 'traffic_light': 0, 'fire_hydrant': 0, 'stop_sign': 0, 'parking_meter': 0}
    total_objs = 0

    for detection in detections:
        labels = detection['labels']
        boxes = detection['boxes']
        scores = detection['scores']

        for label, box, score in zip(labels, boxes, scores):
                if score > score_threshold:
                    xmin,ymin, xmax, ymax = box.tolist()
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)


                    for obj, cat in coco.items():
                        # import pdb; pdb.set_trace()
                        if label == cat:
                            objects[obj] += 1

                    # Draw the bounding box
                    cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

                # Annotate with the class name and score
                    label_text = f"Class {label}, Score {score:.2f}"
                    cv2.putText(dst, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        total_objs = sum(objects.values())

        cv2.putText(dst, f"Number of Objects: {str(total_objs)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    if save_path:
        cv2.imwrite(save_path, dst)  # Save the image
    
    return total_objs, objects, num_lanes


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default='result.png', help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold'
    )
    args = parser.parse_args()
    return args

def get_prediction(lanes, ori_h, ori_w):
    preds = []
    for lane in lanes:
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

def roi(img, vertices):

    mask = np.zeros_like(img)

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, (255,255,255) )
    indices = np.where(mask != (255, 255, 255))
    img[indices] = 0

    return img

fps = 0

def main(args):

    ops =[
            {'type': 'Resize', 'size': [640, 640]},
            {'type': 'ToImageTensor'}, 
            {'type': 'ConvertDtype'}, 
            {'type': 'SanitizeBoundingBox', 'min_size': 1}, 
            {'type': 'ConvertBox', 'out_fmt': 'cxcywh', 'normalize': True}]

    culaneyolo_transforms = Compose(ops)
    postprocessor = RTDETRPostProcessor(num_top_queries= 300)
    cfg = Config.fromfile(args.config)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()
    model.bbox_head.evaluating=True

    culaneyolo_test = CulaneYolo( 
        data_root = cfg.data.val['data_root'],
        data_list = cfg.data.val['data_list'],
        pipeline=cfg.data.val['pipeline'],
        ann_file = 'dataset2/culaneyolo/docker_format/custom_val_docker.json',
        transforms=culaneyolo_transforms,
        test_mode=False,
        return_masks=False,
        remap_mscoco_category=True,
        culaneyolotest=True
    )

    data_loader = DataLoader(
        culaneyolo_test,batch_size=1,shuffle = False,num_workers=4, collate_fn=lambda x: x
    )

    frame, total, fps = 0, 0, 0
    vertices = np.array([[0,0],[1640,0],[1640,450],[0,450]], np.int32)
    Inference = False

    if box=="Inference":
        st.sidebar.info('Click on the ***button*** to run the Inference on the model using the ***pre-defined dataloader***.')
        
        st.subheader("Inference on Dataloader")
        st.write("This uses the pre-defined Dataloader for the inference. The dataloader consists of the Validation images of the CULane Dataset.")

        # st.write("FPS:", st.session_state["FPS"])  # Display the FPS variable
        Inference = st.button("Inference!")

    Stop_Inference = st.button("Stop Inference!")

    st.write("---")

    c1, c2, c3 = st.columns(3)

    sub1 = st.empty()
    fps_cont = st.empty()
    TObj_cont = st.empty()
    lane_cont = st.empty()
    obj_cont = st.empty()
    res_cont = st.empty()
        
    if Inference:

        with torch.no_grad():
            for batch in (data_loader):
                targets= []
                img_metas = []

                for i in range(len(batch)):
                    if (i):
                        img = torch.cat((img,batch[i][0].unsqueeze(0)))
                        img2 = torch.cat((img2,batch[i][3].unsqueeze(0)))

                    else:
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

                img = img.to(device='cuda')
                img2 = img2.to(device='cuda')
                for i2 in range(len(targets)):
                        if (len(targets[i2]['boxes'].shape)==1):
                            targets[i2]['labels']= targets[i2]['labels'].unsqueeze(0)
                        targets[i2]['size'] = torch.tensor([640,640]).to(device='cuda')

                frame +=1
                start = time.time()

                #ye output output sirf
                outputs = model(img=img2, img_metas=img_metas, task="object", targets=targets)
                
                #ye lanes 
                results2= model.forward_test(img=img,img_metas=img_metas)

                end = time.time()
                diff = end - start  
                total = total + diff

                if frame % 10 == 0:
                    print(f"Total Time for {frame} Frames: {total:.2f}s")
                    
                    fps = frame / total
                    fps = "{:.2f}".format(fps)

                # if frame == 100:
                #     import pdb; pdb.set_trace()

                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
                results = postprocessor(outputs, orig_target_sizes)

                lanes = results2[0]['result']['lanes']
                ori_shape = (590,1640)

                preds = get_prediction(lanes, ori_shape[0], ori_shape[1])

                src = img_metas[0]['filename']
                src = cv2.imread(src)

                total_objs, objects, num_lanes = visualize_detections(src=src, preds=preds, detections=results, FPS=fps, save_path='demo/ali.png')
                
                print("   ")
                print(f"FPS for {frame} Frames: ", fps)
                print(f"Number of Lanes: {num_lanes}")
                print("Total Objects Detected:", total_objs)
                print("Objects: ", objects)    

                sub1.subheader("Live Inference Stats!")
                fps_cont.write(f"FPS: {float(fps)}")
                TObj_cont.write(f"Objects: {total_objs}")
                lane_cont.write(f"Lanes: {num_lanes}")
                obj_cont.write(f"All Objects: {objects}")
                res_cont.image("demo/ali.png", caption="Inference Result", width=None, use_column_width="auto")


                if Stop_Inference:
                    Inference = False
                    break

                # st.session_state["FPS"] = fps  # Store the FPS variable in session state

                # import pdb; pdb.set_trace()

# st.markdown(main(parse_args()))

if __name__ == '__main__':
    args = parse_args()
    main(args)