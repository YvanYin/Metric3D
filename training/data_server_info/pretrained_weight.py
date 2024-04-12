db_info={}



db_info['checkpoint']={
    'db_root': 'tbd_weight_root', # Config your weight root!

    # pretrained weight for vit
    'vit_small_reg':  'vit/dinov2_vits14_reg4_pretrain.pth',
    'vit_large_reg':  'vit/dinov2_vitl14_reg4_pretrain.pth',
    'vit_giant2_reg':  'vit/dinov2_vitg14_reg4_pretrain.pth',

    'vit_large': 'vit/dinov2_vitl14_pretrain.pth',

    # pretrained weight for convnext
    'convnext_tiny': 'convnext/convnext_tiny_22k_1k_384.pth',
    'convnext_small': 'convnext/convnext_small_22k_1k_384.pth',
    'convnext_base': 'convnext/convnext_base_22k_1k_384.pth',
    'convnext_large': 'convnext/convnext_large_22k_1k_384.pth',
    'convnext_xlarge': 'convnext/convnext_xlarge_22k_1k_384_ema.pth',
}