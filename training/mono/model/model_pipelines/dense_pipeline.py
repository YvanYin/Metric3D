import torch
import torch.nn as nn
from mono.utils.comm import get_func


class DensePredModel(nn.Module):
    def __init__(self, cfg):
        super(DensePredModel, self).__init__()

        self.encoder = get_func('mono.model.' + cfg.model.backbone.prefix + cfg.model.backbone.type)(**cfg.model.backbone)
        self.decoder = get_func('mono.model.' + cfg.model.decode_head.prefix + cfg.model.decode_head.type)(cfg)
        # try:
        #     decoder_compiled = torch.compile(decoder, mode='max-autotune')
        #     "Decoder compile finished"
        #     self.decoder = decoder_compiled
        # except:
        #     "Decoder compile failed, use default setting"
        #     self.decoder = decoder

        self.training = True
    
    def forward(self, input, **kwargs):
        # [f_32, f_16, f_8, f_4]
        features = self.encoder(input)
        # [x_32, x_16, x_8, x_4, x, ...]
        out = self.decoder(features, **kwargs)
        return out