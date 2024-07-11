"""
Adapted from:
https://github.com/aliyun/conditional-lane-detection/blob/master/mmdet/datasets/culane_dataset.py
"""


import shutil
from pathlib import Path

import torch
import torch.utils.data

import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision import datapoints

from pycocotools import mask as coco_mask
import cv2
import numpy as np
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.utils import get_root_logger
from tqdm import tqdm

from libs.datasets.metrics.culane_metric import eval_predictions
from libs.datasets.pipelines import Compose
from libs.datasets.pipelines import CollectCLRNet
from pycocotools.coco import COCO


@DATASETS.register_module
class CulaneYolo(torchvision.datasets.CocoDetection,CustomDataset):
    """Culane Dataset class."""

    def __init__(
        self,
        data_root,
        ann_file,
        transforms,
        data_list,
        pipeline,
        diff_file=None,
        diff_thr=15,
        test_mode=True,
        y_step=2,
        return_masks=False,
        remap_mscoco_category=True,
        culaneyolotest=False

    ):
        
        super(CulaneYolo, self).__init__(data_root, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = data_root
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self.coco = COCO(self.ann_file)
        self.ids = list((self.coco.imgs.keys()))
        # self.culaneyolotest
        """
        Args:
            data_root (str): Dataset root path.
            data_list (str): Dataset list file path.
            pipeline (List[mmcv.utils.config.ConfigDict]):
                Data transformation pipeline configs.
            test_mode (bool): Test flag.
            y_step (int): Row interval (in the original image's y scale) to sample
                the predicted lanes for evaluation.

        """
        self.culaneyolotest =culaneyolotest
        self.img_prefix = data_root
        self.test_mode = test_mode
        self.ori_w, self.ori_h = 1640, 590
        # read image list
        self.diffs = np.load(diff_file)["data"] if diff_file is not None else []
        self.diff_thr = diff_thr
        self.img_infos, self.annotations, self.mask_paths = self.parse_datalist(
            data_list
        )
        # print(self.mask_paths)
        print(len(self.img_infos), "data are loaded")
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # build data pipeline
        self.pipeline = Compose(pipeline)
        self.result_dir = "tmp"
        self.list_path = data_list
        self.test_categories_dir = str(Path(data_root).joinpath("list/test_split/"))
        self.y_step = y_step

    def parse_datalist(self, data_list):
        """
        Read image data list.
        Args:
            data_list (str): Data list file path.
        Returns:
            List[str]: List of image paths.
        """
        img_infos, annotations, mask_paths = [], [], []
        with open(data_list) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if len(self.diffs) > 0 and self.diffs[i] < self.diff_thr:
                    continue
                img_paths = line.strip().split(" ")
                img_infos.append(img_paths[0].lstrip("/"))
                if not self.test_mode:
                    anno_path = img_paths[0].replace(".jpg", ".lines.txt")
                    annotations.append(anno_path.lstrip("/"))
                    # print(len(img_paths))
                    if len(img_paths) > 1:
                        mask_paths.append(img_paths[1].lstrip("/"))
                    else:
                        mask_paths.append(None)
        return img_infos, annotations, mask_paths

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.img_infos)
    def __getitem__(self, idx):
            img, target = super(CulaneYolo, self).__getitem__(idx)
            #image in PIL and target os dataset[annotations]
            #override attribute

            image_id = self.ids[idx]
            # import pdb; pdb.set_trace()
            # print(image_id)
            target = {'image_id': image_id, 'annotations': target}
            # print(target['image_id'])
            img, target = self.prepare(img, target)
            # print(target['image_id'])

            # ['boxes', 'masks', 'labels']:
            # print("Before xyxy:",target["boxes"])
            if 'boxes' in target:
                target['boxes'] = datapoints.BoundingBox(
                    target['boxes'], 
                    format=datapoints.BoundingBoxFormat.XYXY, 
                    spatial_size=img.size[::-1]) # h w
            # print("after xyxy:",target["boxes"])

            if 'masks' in target:
                target['masks'] = datapoints.Mask(target['masks'])

            if self._transforms is not None:
                img2,target = self._transforms(img,target) #target removed

            imgname = str(Path(self.img_prefix).joinpath(self.img_infos[idx]))
            sub_img_name = self.img_infos[idx]
            img = cv2.imread(imgname)
            ori_shape = img.shape
            kps, id_classes, id_instances = self.load_labels(idx)
            
            results = dict(
            filename=imgname,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=kps,
            id_classes=id_classes,
            id_instances=id_instances,
            img_shape=ori_shape,
            ori_shape=ori_shape,
            gt_masks=None,
            # target = target
        )
            # import pdb; pdb.set_trace()

        #     collect = CollectCLRNet(keys=["img"],
        # meta_keys=[
        #     "filename",
        #     "sub_img_name",
        #     "ori_shape",
        #     "img_shape",
        #     # "img_norm_cfg",
        #     "ori_shape",
        #     "img_shape",
        #     "gt_points",
        #     "gt_masks",
        #     "lanes",
        # ],
# )
            
            # import pdb; pdb.set_trace()
            # print(bool(self.mask_paths[0]))
            # print(self.test_mode)
            if self.mask_paths[0]:
                results["gt_masks"] = self.load_mask(idx)
            pipelined_results = self.pipeline(results)
            # import pdb; pdb.set_trace()

            img = pipelined_results['img'].data

            # img = pipelined_results['img']
            # img_metas = pipelined_results['img_metas']
            img_metas = pipelined_results['img_metas'].data
            if(not self.culaneyolotest):
                img_metas['gt_masks'] = img_metas['gt_masks'].data
            target['boxes'] = target['boxes'].squeeze(0)
            
            # print(target['image_id'])
            return img,img_metas,target,img2
            # return results

            # return img, target

    # def prepare_train_img(self, idx):
    #     """
    #     Read and process the image through the transform pipeline for training.
    #     Args:
    #         idx (int): Data index.
    #     Returns:
    #         dict: Pipeline results containing
    #             'img' and 'img_meta' data containers.
    #     """
    #     imgname = str(Path(self.img_prefix).joinpath(self.img_infos[idx]))
    #     sub_img_name = self.img_infos[idx]
    #     img = cv2.imread(imgname)
    #     ori_shape = img.shape
    #     kps, id_classes, id_instances = self.load_labels(idx)
    #     results = dict(
    #         filename=imgname,
    #         sub_img_name=sub_img_name,
    #         img=img,
    #         gt_points=kps,
    #         id_classes=id_classes,
    #         id_instances=id_instances,
    #         img_shape=ori_shape,
    #         ori_shape=ori_shape,
    #         gt_masks=None,
    #     )
    #     if self.mask_paths[0]:
    #         results["gt_masks"] = self.load_mask(idx)
    #     return self.pipeline(results)

    def prepare_test_img(self, idx):
        """
        Read and process the image through the transform pipeline for test.
        Args:
            idx (int): Data index.
        Returns:
            dict: Pipeline results containing
                'img' and 'img_meta' data containers.
        """
        img_name = str(Path(self.img_prefix).joinpath(self.img_infos[idx]))
        sub_img_name = self.img_infos[idx]
        img = cv2.imread(img_name)
        ori_shape = img.shape
        results = dict(
            filename=img_name,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=[],
            id_classes=[],
            id_instances=[],
            img_shape=ori_shape,
            ori_shape=ori_shape,
        )
        return self.pipeline(results)

    def load_mask(self, idx):
        """
        Read a segmentation mask for training.
        Args:
            idx (int): Data index.
        Returns:
            numpy.ndarray: segmentation mask.
        """
        maskname = str(Path(self.img_prefix).joinpath(self.mask_paths[idx]))
        mask = cv2.imread(maskname, cv2.IMREAD_UNCHANGED)
        return mask

    def load_labels(self, idx):
        """
        Read a ground-truth lane from an annotation file.
        Args:
            idx (int): Data index.
        Returns:
            List[list]: list of lane point lists.
            list: class id (=1) for lane instances.
            list: instance id (start from 1) for lane instances.
        """
        anno_dir = str(Path(self.img_prefix).joinpath(self.annotations[idx]))
        shapes = []
        with open(anno_dir, "r") as anno_f:
            lines = anno_f.readlines()
            for line in lines:
                coords = []
                coords_str = line.strip().split(" ")
                for i in range(len(coords_str) // 2):
                    coord_x = float(coords_str[2 * i])
                    coord_y = float(coords_str[2 * i + 1])
                    coords.append(coord_x)
                    coords.append(coord_y)
                if len(coords) > 3:
                    shapes.append(coords)
        id_classes = [1 for i in range(len(shapes))]
        id_instances = [i + 1 for i in range(len(shapes))]
        return shapes, id_classes, id_instances

    def evaluate(self, results, metric="F1", logger=None):
        """
        Write prediction to txt files for evaluation and
        evaluate them with labels.
        Args:
            results (List[dict]): All inference results containing:
                result (dict): contains 'lanes' and 'scores'.
                meta (dict): contains meta information.
            metric (str): Metric type to evaluate. (not used)
        Returns:
            dict: Evaluation result dict containing
                F1, precision, recall, etc. on the specified IoU thresholds.

        """
        for result in tqdm(results):
            lanes = result["result"]["lanes"]
            dst_path = (
                Path(self.result_dir)
                .joinpath(result["meta"]["sub_img_name"])
                .with_suffix(".lines.txt")
            )
            dst_path.parents[0].mkdir(parents=True, exist_ok=True)
            with open(str(dst_path), "w") as f:
                output = self.get_prediction_string(lanes)
                if len(output) > 0:
                    print(output, file=f)

        results = eval_predictions(
            self.result_dir,
            self.img_prefix,
            self.list_path,
            self.test_categories_dir,
            logger=get_root_logger(log_level="INFO"),
        )
        shutil.rmtree(self.result_dir)
        return results

    def get_prediction_string(self, lanes):
        """
        Convert lane instance structure to prediction strings.
        Args:
            lanes (List[Lane]): List of lane instances in `Lane` structure.
        Returns:
            out_string (str): Output string.
        """
        ys = np.arange(0, self.ori_h, self.y_step) / self.ori_h
        out = []
        for lane in lanes:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.ori_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.ori_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            if len(lane_xs) < 2:
                continue
            lane_str = " ".join(
                ["{:.5f} {:.5f}".format(x, y) for x, y in zip(lane_xs, lane_ys)]
            )
            if lane_str != "":
                out.append(lane_str)
        return "\n".join(out) if len(out) > 0 else ""

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
         
        image_id = str(image_id)
        # import pdb; pdb.set_trace()
        # print(image_id)
        leftover = image_id.replace(image_id[5:], "")
        if (len(leftover)+len(str(int(image_id[5:])))<len(image_id)):
            leftover= leftover + '0'
        leftover = torch.tensor(int(leftover))
        image_id = int(image_id[5:])
        # print(image_id)
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_mscoco_category:
            classes = [mscoco_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]
            
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])
        target["leftover"]= leftover
    
        return image, target


mscoco_category2name = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    # 15: 'bench',
    # 16: 'bird',
    # 17: 'cat',
    # 18: 'dog',
    # 19: 'horse',
    # 20: 'sheep',
    # 21: 'cow',
    # 22: 'elephant',
    # 23: 'bear',
    # 24: 'zebra',
    # 25: 'giraffe',
    # 27: 'backpack',
    # 28: 'umbrella',
    # 31: 'handbag',
    # 32: 'tie',
    # 33: 'suitcase',
    # 34: 'frisbee',
    # 35: 'skis',
    # 36: 'snowboard',
    # 37: 'sports ball',
    # 38: 'kite',
    # 39: 'baseball bat',
    # 40: 'baseball glove',
    # 41: 'skateboard',
    # 42: 'surfboard',
    # 43: 'tennis racket',
    # 44: 'bottle',
    # 46: 'wine glass',
    # 47: 'cup',
    # 48: 'fork',
    # 49: 'knife',
    # 50: 'spoon',
    # 51: 'bowl',
    # 52: 'banana',
    # 53: 'apple',
    # 54: 'sandwich',
    # 55: 'orange',
    # 56: 'broccoli',
    # 57: 'carrot',
    # 58: 'hot dog',
    # 59: 'pizza',
    # 60: 'donut',
    # 61: 'cake',
    # 62: 'chair',
    # 63: 'couch',
    # 64: 'potted plant',
    # 65: 'bed',
    # 67: 'dining table',
    # 70: 'toilet',
    # 72: 'tv',
    # 73: 'laptop',
    # 74: 'mouse',
    # 75: 'remote',
    # 76: 'keyboard',
    # 77: 'cell phone',
    # 78: 'microwave',
    # 79: 'oven',
    # 80: 'toaster',
    # 81: 'sink',
    # 82: 'refrigerator',
    # 84: 'book',
    # 85: 'clock',
    # 86: 'vase',
    # 87: 'scissors',
    # 88: 'teddy bear',
    # 89: 'hair drier',
    # 90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}