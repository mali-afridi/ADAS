model = dict(
    type="CLRerNet_Transformer2",
    backbone2=dict(
        type="DLANet",
        dla="dla34",
        pretrained=False,
    ),
    # backbone = dict(
    #     type="PResNet",
    #     depth= 50,
    #     variant= 'd',
    #     freeze_at= 0,
    #     return_idx= [1, 2, 3],
    #     num_stages=4,
    #     freeze_norm= True,
    #     pretrained= True,
    # ),
    backbone =dict(
        type="DLANet",
        dla="dla34",
        pretrained=False,
    ),
    # backbone = dict(),
    neck=dict(
        type="CLRerNetFPN",
        in_channels=[128, 256, 512],
        # in_chlanesannels=[512, 1024, 2048],
        out_channels=64,
        num_outs=3,
    ),
    # neck2 = None,
    encoder = dict(
        type = "HybridEncoder",
        in_channels =[128, 256, 512],
        # in_channels= [512, 1024, 2048],
        feat_strides= [8, 16, 32],

        # intra
        hidden_dim= 256,
        use_encoder_idx= [2],
        num_encoder_layers= 1,
        nhead= 8,
        dim_feedforward= 1024,
        dropout= 0.,
        enc_act= 'gelu',
        pe_temperature= 10000,
        
        # cross
        expansion= 1.0,
        depth_mult= 1,
        act= 'silu',

        # eval
        eval_spatial_size= [640, 640]
  
    ),
    bbox_head=dict(
        type="CLRerHead",
        anchor_generator=dict(
            type="CLRerNetAnchorGenerator",
            num_priors=192,
            num_points=72,
        ),
        img_w=800, #800
        img_h=320, #320
        prior_feat_channels=64,
        fc_hidden_dim=64,
        num_fc=2,
        refine_layers=3,
        sample_points=36,
        attention=dict(type="ROIGather"),
        loss_cls=dict(type="KorniaFocalLoss", alpha=0.25, gamma=2, loss_weight=2.0),
        loss_bbox=dict(type="SmoothL1Loss", reduction="none", loss_weight=0.2),
        loss_iou=dict(
            type="LaneIoULoss",
            lane_width=7.5 / 800,
            loss_weight=4.0,
        ),
        # loss_seg = None,
        loss_seg=dict(
            type="CLRNetSegLoss",
            loss_weight=1.0,
            num_classes=5,  # 4 lanes + 1 background
            ignore_label=255,
            bg_weight=0.4,
        ),
        evaluating=False
    ),
    # head2 = None,
    decoder = dict(
        type = "RTDETRTransformer",
        feat_channels= [256, 256, 256],
        feat_strides= [8, 16, 32],
        hidden_dim= 256,
        num_levels= 3,

        num_queries= 300,

        num_decoder_layers= 6,
        num_denoising= 100,
        
        eval_idx= -1,
        eval_spatial_size= [640, 640]
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="DynamicTopkAssigner",
            max_topk=4,
            min_topk=1,
            cost_combination=1,
            cls_cost=dict(type="FocalCost", weight=1.0),
            reg_cost=dict(type="DistanceCost", weight=0.0),
            iou_dynamick=dict(
                type="LaneIoUCost",
                lane_width=7.5 / 800,
                use_pred_start_end=False,
                use_giou=True,
            ),
            iou_cost=dict(
                type="LaneIoUCost",
                lane_width=30 / 800, #800
                use_pred_start_end=True,
                use_giou=True,
            ),
        )
    ),
    test_cfg=dict(
        # conf threshold is obtained from cross-validation
        # of the train set. The following value is
        # for CLRerNet w/ DLA34 & EMA model.
        conf_threshold=0.41,
        use_nms=True,
        as_lanes=True,
        nms_thres=50,
        nms_topk=4,
        ori_img_w=1640,
        ori_img_h=590,
        cut_height=270,
    ),
)
