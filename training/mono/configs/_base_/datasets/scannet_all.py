# dataset settings
# data will resized/cropped to the canonical size, refer to ._data_base_.py

ScanNetAll_dataset=dict(
    lib = 'ScanNetDataset',
    data_root = 'data/public_datasets',
    data_name = 'ScanNetAll',
    transfer_to_canonical = True,
    metric_scale = 1000.0,
    original_focal_length = 1165.371094,
    original_size = (968, 1296),
    data_type='denselidar',
    data = dict(
    # configs for the training pipeline
    train=dict(
        anno_path='ScanNet/annotations/test.json',
        sample_ratio = 1.0,
        sample_size = -1,
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='ResizeCanonical', ratio_range=(0.9, 1.4)),
                  dict(type='RandomCrop', 
                       crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                       crop_type='rand', 
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                 dict(type='RandomEdgeMask',
                         mask_maxsize=50, 
                         prob=0.2, 
                         rgb_invalid=[0,0,0], 
                         label_invalid=-1,),
                  dict(type='RandomHorizontalFlip', 
                       prob=0.4),
                  dict(type='PhotoMetricDistortion', 
                       to_gray_prob=0.2,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.1),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],),

    # configs for the training pipeline
    val=dict(
        anno_path='ScanNet/annotations/test.json',
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='ResizeCanonical', ratio_range=(1.0, 1.0)),
                  dict(type='RandomCrop', 
                       crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                       crop_type='center', 
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_ratio = 1.0,
        sample_size = 20,),
    # configs for the training pipeline
    test=dict(
        anno_path='ScanNet/annotations/test.json',
        pipeline=[dict(type='BGR2RGB'),
               #   dict(type='LiDarResizeCanonical', ratio_range=(1.0, 1.0)),
                  dict(type='ResizeKeepRatio', 
                       resize_size=(512, 960),
                       ignore_label=-1, 
                       padding=[0, 0, 0]),
               #    dict(type='RandomCrop', 
               #         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
               #         crop_type='center', 
               #         ignore_label=-1, 
               #         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_ratio = 1.0,
        sample_size = -1,),
     ),
)