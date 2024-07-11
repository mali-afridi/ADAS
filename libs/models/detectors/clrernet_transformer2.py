from mmdet.models.builder import DETECTORS,BACKBONES, NECKS, HEADS, build_head,build_neck,build_backbone
from mmdet.models.detectors.single_stage import SingleStageDetector
import numpy as np 
import torch.nn.functional as F 

# from mmdet.models.builder import 
@DETECTORS.register_module()
class CLRerNet_Transformer2(SingleStageDetector):
    def __init__(
        self,
        backbone,
        backbone2,
        neck,
        encoder,
        bbox_head,
        decoder,
        train_cfg,
        test_cfg,
        pretrained=None,
        init_cfg=None,
        multi_scale = None,
        # multi_scale= [480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800],
    ):
        """CLRerNet detector."""
        super(CLRerNet_Transformer2, self).__init__(
            backbone2, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg
        )
        # super(CLRerNet, self).__init__()
        # import pdb; pdb.set_trace()
        self.backbone2 = build_backbone(backbone2)
        self.neck=build_neck(neck)
        self.bbox_head = build_head(bbox_head)
        self.multi_scale = multi_scale
        # self.extract_features = self.backbone2

        # if neck is not None:
        #         self.neck.add_module('bbox_head', self.bbox_head)
        # self.neck2=build_neck(neck2)
        if backbone is not None:
            self.backbone = build_backbone(backbone)
        if encoder is not None:
             self.encoder = build_neck(encoder)
    
        if decoder is not None:
            self.decoder = build_head(decoder)
            # if neck2 is not None:
            #     self.neck2.add_module('head2', self.head2)

    def forward_train(self, img, img_metas,task = "",targets=None, **kwargs):
        #targets from rtdetr
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas) # returns loss from bbox_head
        # super(SingleStageDetector, self)
        # x = self.extract_feat(img) #returns features from backbone followed by neck
        # #lane_seg
        # losses_clrnet = self.bbox_head.forward_train(x, img_metas) # returns loss from bbox_head
        # #obj_det
        # return losses_clrnet
        #optional rescaling
        # if self.multi_scale and self.training:
        #     sz = np.random.choice(self.multi_scale)
        #     x = F.interpolate(img, size=[sz, sz])
        # print(x.shape)
        if task =="object":
            # import pdb; pdb.set_trace()
            obj_features = self.backbone(img)
            if self.encoder is not None:
                obj_neck = self.encoder(obj_features[1:])
            if self.decoder is not None:
                decoder_obj = self.decoder(obj_neck, targets)
            return decoder_obj
            

        if task == "lane":
            lane_features = self.backbone2(img)
            fpn_features = self.neck(lane_features)
            losses_clrnet = self.bbox_head.forward_train(fpn_features, img_metas)
            return losses_clrnet

        #this neck1 has a pointer which changes input too
        # print("passed through neck1")
        # print(len(x[1:]))
         #encoder
        
        # print("passed thorugh neck2")
        
        # print("passed through head1")
        # print(len(x2))
        
        # print("passed through transformer decoder")

        


    def forward_test(self, img, img_metas, **kwargs):
        """
        
        Single-image test without augmentation.
        Args:
            img (torch.Tensor): Input image tensor of shape (1, 3, height, width).
            img_metas (List[dict]): Meta dict containing image information.
        Returns:
            result_dict (List[dict]): Single-image result containing prediction outputs and
             img_metas as 'result' and 'metas' respectively.
        """
        # import pdb; pdb.set_trace()
        assert (
            img.shape[0] == 1 and len(img_metas) == 1
        ), "Only single-image test is supported."
        img_metas[0]["batch_input_shape"] = tuple(img.size()[-2:])

        # x = self.extract_feat(img)
        x = self.backbone2(img)
        x = self.neck(x)
        # import pdb; pdb.set_trace()
        output = self.bbox_head.simple_test(x)
        result_dict = {
            "result": output,
            "meta": img_metas[0],
        }
        return [result_dict]  # assuming batch size is 1
