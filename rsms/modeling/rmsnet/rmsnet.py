# -*- coding: utf-8 -*-
"""
@author: S. Cai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsms.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from rsms.modeling.rmsnet.segformer_encoder import mit_b2
from rsms.modeling.rmsnet.samixer import SAMixerHead  # use SAMixer

# --------------------------------------
from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation

class RMSNet(nn.Module):
    """
    'encoder_id' denotes the type of Mix-Transformer, default: 2, means mit_b2
    """

    def __init__(
        self,
        num_classes=20,
        backbone="mit_b2",
        encoder_id=2,
        sync_bn=True,
        freeze_bn=False,
    ):
        super(RMSNet, self).__init__()

        if sync_bn == True:
            norm_layer = SynchronizedBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        # self.backbone = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b2")
        # self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
        # self.backbone = SegformerForSemanticSegmentation.from_pretrained(
        #     "nvidia/segformer-b2-finetuned-ade-512-512"
        # )

        self.backbone = mit_b2()
        self.decoder = SAMixerHead(
            in_channels=[64, 128, 320, 512],
            feature_strides=[4, 8, 16, 32],
            embedding_dim=768,
            norm_layer=norm_layer,
            num_classes=20,
        )

        self.freeze_bn = freeze_bn

    # if there is no se_loss
    def forward(self, input):

        features = self.backbone(input)
        seg_mask = self.decoder(features)

        seg_mask = F.interpolate(
            seg_mask,
            size=input.size()[2:],
            mode="bicubic",
            align_corners=True,
            antialias=True,
        )

        return seg_mask

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()
