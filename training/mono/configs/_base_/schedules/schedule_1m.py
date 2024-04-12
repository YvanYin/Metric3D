optimizer = dict(
    type='SGD', 
    encoder=dict(lr=0.01, ),
    decoder=dict(lr=0.01, ),
)
# learning policy
lr_config = dict(policy='poly',) #dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)


