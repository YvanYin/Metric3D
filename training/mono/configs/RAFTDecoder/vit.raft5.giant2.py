_base_=['../_base_/losses/all_losses.py',
       '../_base_/models/encoder_decoder/dino_vit_small_reg.dpt_raft.py',

       '../_base_/datasets/ddad.py',
       '../_base_/datasets/_data_base_.py',
       '../_base_/datasets/argovers2.py',
       '../_base_/datasets/cityscapes.py',
       '../_base_/datasets/drivingstereo.py',
       '../_base_/datasets/dsec.py',
       '../_base_/datasets/lyft.py',
       '../_base_/datasets/mapillary_psd.py',
       '../_base_/datasets/diml.py',
       '../_base_/datasets/taskonomy.py',
       '../_base_/datasets/uasol.py',
       '../_base_/datasets/pandaset.py',
       '../_base_/datasets/waymo.py',

       '../_base_/default_runtime.py',
       '../_base_/schedules/schedule_1m.py',
       
       '../_base_/datasets/hm3d.py',
       '../_base_/datasets/matterport3d.py',
       '../_base_/datasets/replica.py',
       '../_base_/datasets/vkitti.py',
       ]

import numpy as np
model=dict(
    decode_head=dict(
        type='RAFTDepthNormalDPT5',
        iters=8,
        n_downsample=2,
        detach=False,
    ),
)

# loss method
losses=dict(
    decoder_losses=[
        dict(type='VNLoss', sample_ratio=0.2, loss_weight=1.0),   
        dict(type='GRUSequenceLoss', loss_weight=0.5, loss_gamma=0.9, stereo_sup=0.0),     
        dict(type='SkyRegularizationLoss', loss_weight=0.001, sample_ratio=0.4, regress_value=200, normal_regress=[0, 0, -1]),
        dict(type='HDNRandomLoss', loss_weight=0.5, random_num=10),
        dict(type='HDSNRandomLoss', loss_weight=0.5, random_num=20, batch_limit=4),
        dict(type='PWNPlanesLoss', loss_weight=1),
        dict(type='NormalBranchLoss', loss_weight=1.5, loss_fn='NLL_ours_GRU'),
        dict(type='DeNoConsistencyLoss', loss_weight=0.01, loss_fn='CEL', scale=2, depth_detach=True)
    ],
    gru_losses=[
        dict(type='SkyRegularizationLoss', loss_weight=0.001, sample_ratio=0.4, regress_value=200, normal_regress=[0, 0, -1]),
    ],
)

data_array = [
     # Outdoor 1
    [
         dict(UASOL='UASOL_dataset'), #13.6w
        dict(Cityscapes_trainextra='Cityscapes_dataset'), #1.8w
        dict(Cityscapes_sequence='Cityscapes_dataset'), #13.5w
          dict(DIML='DIML_dataset'), # 12.2w
         dict(Waymo='Waymo_dataset'), # 99w
    ], 
     # Outdoor 2
    [
          dict(DSEC='DSEC_dataset'),
          dict(Mapillary_PSD='MapillaryPSD_dataset'), # 74.2w 
         dict(DrivingStereo='DrivingStereo_dataset'), # 17.6w
         dict(Argovers2='Argovers2_dataset'), # 285.6w
    ],
     # Outdoor 3
    [
          dict(Lyft='Lyft_dataset'), #15.8w
        dict(DDAD='DDAD_dataset'), #7.4w
        dict(Pandaset='Pandaset_dataset'), #3.8w
        dict(Virtual_KITTI='VKITTI_dataset'), # 3.7w # syn
    ],
     #Indoor 1
    [
         dict(Replica='Replica_dataset'), # 5.6w # syn
         dict(Replica_gso='Replica_dataset'), # 10.7w # syn
         dict(Hypersim='Hypersim_dataset'), # 2.4w
         dict(ScanNetAll='ScanNetAll_dataset'),
    ],
     # Indoor 2
    [
          dict(Taskonomy='Taskonomy_dataset'), #447.2w
        dict(Matterport3D='Matterport3D_dataset'), #14.4w
        dict(HM3D='HM3D_dataset'), # 200w, very noisy, sampled some data
    ],
]



# configs of the canonical space
data_basic=dict(
    canonical_space = dict(
        # img_size=(540, 960),
        focal_length=1000.0,
    ),
    depth_range=(0, 1),
    depth_normalize=(0.1, 200),
#     crop_size=(544, 1216),
#     crop_size = (544, 992),
    crop_size = (616, 1064),  # %28 = 0
) 

log_interval = 100
acc_batch = 1
# online evaluation
# evaluation = dict(online_eval=True, interval=1000, metrics=['abs_rel', 'delta1', 'rmse'], multi_dataset_eval=True)
interval = 40000
evaluation = dict(
    online_eval=False, 
    interval=interval, 
    metrics=['abs_rel', 'delta1', 'rmse', 'normal_mean', 'normal_rmse', 'normal_a1'], 
    multi_dataset_eval=True,
    exclude=['DIML_indoor', 'GL3D', 'Tourism', 'MegaDepth'],
)

# save checkpoint during training, with '*_AMP' is employing the automatic mix precision training
checkpoint_config = dict(by_epoch=False, interval=interval)
runner = dict(type='IterBasedRunner_AMP', max_iters=800010)

# optimizer
optimizer = dict(
    type='AdamW', 
#     encoder=dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6),
    encoder=dict(lr=8e-6, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-6),
    decoder=dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6),
    #strict_match=True
)
# schedule
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=1e-6,
                 power=0.9, min_lr=1e-6, by_epoch=False)

batchsize_per_gpu = 3
thread_per_gpu = 1

Argovers2_dataset=dict(
    data = dict(
    train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.85, 1.15),
                         is_lidar=True),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
Cityscapes_dataset=dict(
    data = dict(
    train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         ratio_range=(0.85, 1.15),
                         is_lidar=False),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
DIML_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         ratio_range=(0.85, 1.15),
                         is_lidar=False),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
Lyft_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.85, 1.15),
                         is_lidar=True),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
DDAD_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.85, 1.15),
                         is_lidar=True),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
     #    sample_size = 1200,
    ),
    ))
DSEC_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.85, 1.15),
                         is_lidar=True),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
DrivingStereo_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         ratio_range=(0.85, 1.15),
                         is_lidar=False),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
MapillaryPSD_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.85, 1.15),
                         is_lidar=True),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
Pandaset_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.85, 1.15),
                         is_lidar=True),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
Taskonomy_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.85, 1.15),
                         is_lidar=False),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
UASOL_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.85, 1.15),
                         is_lidar=False),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
Waymo_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.85, 1.15),
                         is_lidar=True),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
Matterport3D_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.85, 1.15),
                         is_lidar=False),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
Replica_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.85, 1.15),
                         is_lidar=False),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
VKITTI_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.85, 1.15),
                         is_lidar=False),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
HM3D_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.75, 1.3),
                         is_lidar=False),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
BlendedMVG_omni_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.75, 1.3),
                         is_lidar=False),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
    ),
    ))
ScanNetAll_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.85, 1.15),
                         is_lidar=False),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))
Hypersim_dataset=dict(
    data = dict(
        train=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomResize',
                         prob=0.5,
                         ratio_range=(0.85, 1.15),
                         is_lidar=False),
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
                       to_gray_prob=0.1,
                       distortion_prob=0.1,),
                  dict(type='Weather',
                       prob=0.05),
                  dict(type='RandomBlur', 
                       prob=0.05),
                  dict(type='RGBCompresion', prob=0.1, compression=(0, 40)),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        #sample_size = 10000,
    ),
    val=dict(
        pipeline=[dict(type='BGR2RGB'),
                  dict(type='LabelScaleCononical'),
                  dict(type='RandomCrop', 
                         crop_size=(0,0), # crop_size will be overwriteen by data_basic configs
                         crop_type='center', 
                         ignore_label=-1, 
                         padding=[0, 0, 0]),
                  dict(type='ToTensor'),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                 ],
        sample_size = 1200,
    ),
    ))