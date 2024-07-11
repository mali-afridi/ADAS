dataset_type = "CulaneDataset"
# classes = ('person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign', 'parking meter',)
data_root = "data/CULane Complete/CULane"
# data_root = "/home/ali/Desktop/FYP/fusion/data/CULane Complete/CULane"
# crop_bbox = [0, 540, 1920, 1080]
# crop_bbox = [0, 270, 1600, 575]

crop_bbox = [0, 270, 1640, 590]
img_scale = (800, 320)
img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=False)
compose_cfg = dict(bboxes=False, keypoints=True, masks=True)


# data pipeline settings
train_al_pipeline = [
    dict(type="Compose", params=compose_cfg),
    dict(
        type="Crop",
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1,
    ),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
    dict(type="HorizontalFlip", p=0.5),
    dict(type="ChannelShuffle", p=0.1),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=0.04,
        contrast_limit=0.15,
        p=0.6,
    ),
    dict(
        type="HueSaturationValue",
        hue_shift_limit=(-10, 10),
        sat_shift_limit=(-10, 10),
        val_shift_limit=(-10, 10),
        p=0.7,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="MotionBlur", blur_limit=5, p=1.0),
            dict(type="MedianBlur", blur_limit=5, p=1.0),
        ],
        p=0.2,
    ),
    dict(
        type="IAAAffine",
        scale=(0.8, 1.2),
        rotate=(-10.0, 10.0),  # this sometimes breaks lane sorting
        translate_percent=0.1,
        p=0.7,
    ),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]

val_al_pipeline = [
    dict(type="Compose", params=compose_cfg),
    dict(
        type="Crop",
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1,
    ),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]

train_pipeline = [
    dict(type="albumentation", pipelines=train_al_pipeline),
    # dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),

    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectCLRNet",
        keys=["img"],
        meta_keys=[
            "filename",
            "sub_img_name",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "ori_shape",
            "img_shape",
            "gt_points",
            "gt_masks",
            "lanes",
        ],
    ),
]

val_pipeline = [
    dict(type="albumentation", pipelines=val_al_pipeline),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectCLRNet",
        keys=["img"],
        meta_keys=[
            "filename",
            "sub_img_name",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
        ],
    ),
]


data = dict(
    samples_per_gpu=8,  # medium
    workers_per_gpu=8,

    train=dict(
        type=dataset_type,
        data_root=data_root,
        # classes=classes,
        data_list=data_root + "/list/train_gt.txt",
        # ann_file="data/CULane Complete/CULane/list/frame_train_docker.json",
        diff_file=data_root + "/list/train_diffs.npz",
        
        diff_thr=15,
        pipeline=train_pipeline,
        # remap_mscoco_category=True,
        test_mode=False,
        # return_masks = False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        # classes=classes,
        # ann_file='data/CULane Complete/CULane/list/custom_val_docker.json',
        data_list=data_root + "/list/val.txt",
        pipeline=val_pipeline,
        test_mode=True,
        # return_masks = False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        # classes=classes,
        # ann_file='data/CULane Complete/CULane/list/custom_test_docker.json',
        data_list=data_root + "/list/isb.txt",
        pipeline=val_pipeline,
        test_mode=True,
        # return_masks = False,
    ),
)
