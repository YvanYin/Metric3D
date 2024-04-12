_base_=['../_base_/losses/all_losses.py',
       '../_base_/models/encoder_decoder/dino_vit_large_reg.dpt_raft.py',

       '../_base_/datasets/eth3d.py',
       '../_base_/datasets/_data_base_.py',

       '../_base_/default_runtime.py',
       '../_base_/schedules/schedule_1m.py'
       ]

import numpy as np

model = dict(
    decode_head=dict(
        type='RAFTDepthNormalDPT5',
        iters=8,
        n_downsample=2,
        detach=False,
    )
)

# model settings
find_unused_parameters = True



# data configs, some similar data are merged together
data_array = [
    # group 1
    [
        dict(ETH3D='ETH3D_dataset'), #447.2w
    ],
]
data_basic=dict(
    canonical_space = dict(
        # img_size=(540, 960),
        focal_length=1000.0,
    ),
    depth_range=(0, 1),
    depth_normalize=(0.1, 200),# (0.3, 160),
    crop_size = (1120, 2016),
    clip_depth_range=(0.1, 200),
    vit_size=(616,1064),
) 

# indoor (544, 928), outdoor: (768, 1088)
test_metrics = ['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3', 'normal_mean', 'normal_rmse', 'normal_a1']
ETH3D_dataset=dict(
    data = dict(
    test=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='ResizeKeepRatio', 
                    #    resize_size=(512, 512), #(768, 1088), #(768, 1120), # (768, 1216), #(768, 1024), # (768, 1216),  #(768, 1312), # (512, 512)
                       resize_size=(616,1064),
                    #    resize_size=(1120, 2016),
                       ignore_label=-1, 
                       padding=[0,0,0]),
                #   dict(type='RandomCrop',
                #        crop_size=(0,0),
                #        crop_type='center',
                #        ignore_label=-1,
                #        padding=[0,0,0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_ratio = 1.0,
        sample_size = -1,
     ),
    ))
