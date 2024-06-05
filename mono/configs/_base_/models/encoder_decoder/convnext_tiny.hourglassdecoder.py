# model settings
_base_ = ['../backbones/convnext_tiny.py',]
model = dict(
    type='DensePredModel',
    decode_head=dict(
        type='HourglassDecoder',
        in_channels=[96, 192, 384, 768],
        decoder_channel=[64, 64, 128, 256],
        prefix='decode_heads.'),
)
