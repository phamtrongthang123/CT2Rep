import numpy as np
import torch.nn as nn
from modules.encoder_decoder import EncoderDecoder
from modules.tokenizers import Tokenizer
from modules.visual_extractor import VisualExtractor
from omegaconf import OmegaConf
from ctvit import CTViT


class CT2RepModel(nn.Module):
    def __init__(self, encoder_decoder_config, tokenizer_config):
        super(CT2RepModel, self).__init__()
        self.tokenizer = Tokenizer(**tokenizer_config)

        model = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=24,
            temporal_patch_size=12,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8,
        )

        self.visual_extractor = VisualExtractor(model)
        self.encoder_decoder = EncoderDecoder(OmegaConf.create(encoder_decoder_config), self.tokenizer)
        self.forward = self.forward_ct2rep

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def forward_ct2rep(self, images, targets=None, mode="train"):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == "train":
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode="forward")
        elif mode == "sample":
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode="sample")
        else:
            raise ValueError
        return output
