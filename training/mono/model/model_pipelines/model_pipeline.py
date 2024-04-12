import torch
import torch.nn as nn
from mono.utils.comm import get_func


class EncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super(EncoderDecoder, self).__init__()

        self.encoder = get_func('mono.model.' + cfg.model.backbone.prefix + cfg.model.backbone.type)(**cfg.model.backbone)
        self.decoder = get_func('mono.model.' + cfg.model.decode_head.prefix + cfg.model.decode_head.type)(cfg)

        self.depth_out_head = DepthOutHead(method=cfg.model.depth_out_head.method, **cfg)
        self.training = True
    
    def forward(self, input, **kwargs):
        # [f_32, f_16, f_8, f_4]
        features = self.encoder(input)
        # [x_32, x_16, x_8, x_4, x, ...]
        decode_list = self.decoder(features)

        pred, conf, logit, bins_edges = self.depth_out_head([decode_list[4], ])

        auxi_preds = None
        auxi_logits = None
        out = dict(
            prediction=pred[0], 
            confidence=conf[0], 
            pred_logit=logit[0],
            auxi_pred=auxi_preds, 
            auxi_logit_list=auxi_logits,
            bins_edges=bins_edges[0],
        )
        return out