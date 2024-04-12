"""
There are multiple losses can be applied. 

dict(type='GradientLoss_Li', scale_num=4, loss_weight=1.0),
dict(type='VNLoss', sample_ratio=0.2, loss_weight=1.0),
dict(type='SilogLoss', variance_focus=0.5, loss_weight=1.0),
dict(type='WCELoss', loss_weight=1.0, depth_normalize=(0.1, 1), bins_num=200)
dict(type='RegularizationLoss', loss_weight=0.1)
dict(type='EdgeguidedRankingLoss', loss_weight=1.0)
Note that out_channel and depth_normalize will be overwriten by configs in data_basic. 
"""

# loss_decode=[dict(type='VNLoss', sample_ratio=0.2, loss_weight=1.0),
#              #dict(type='SilogLoss', variance_focus=0.5, loss_weight=1.0),
#              dict(type='WCELoss', loss_weight=1.0, depth_normalize=(0, 0), out_channel=0)]

# loss_auxi = [#dict(type='WCELoss', loss_weight=1.0, depth_normalize=(0.1, 1), out_channel=200),
#             ]
losses=dict(
    decoder_losses=[
        dict(type='VNLoss', sample_ratio=0.2, loss_weight=1.0),
        dict(type='WCELoss', loss_weight=1.0, depth_normalize=(0, 0), out_channel=0),
    ],
    auxi_losses=[],
    pose_losses=[],
)
