# distributed training configs, if  dist_url == 'env://'('tcp://127.0.0.1:6795'), nodes related configs should be set in the shell
dist_params = dict(port=None, backend='nccl', dist_url='env://')

log_name = 'tbd'
log_file = 'out.log'

load_from = None
resume_from = None

#workflow = [('train', 1)]
cudnn_benchmark = True
log_interval = 20

use_tensorboard = True

evaluation = dict(online_eval=True, interval=1000, metrics=['abs_rel', 'delta1'])
checkpoint_config = dict(by_epoch=False, interval=16000)


# runtime settings, IterBasedRunner or EpochBasedRunner, e.g. runner = dict(type='EpochBasedRunner', max_epoches=100)
runner = dict(type='IterBasedRunner', max_iters=160000)

test_metrics = ['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3', 'rmse_log', 'log10', 'sq_rel']