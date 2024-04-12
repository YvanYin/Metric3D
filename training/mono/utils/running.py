import os
import torch
import torch.nn as nn
from mono.utils.comm import main_process
import copy
import inspect
import logging
import glob

class LrUpdater():
    """Refer to LR Scheduler in MMCV.
    Args:
        @by_epoch (bool): LR changes epoch by epoch
        @warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        @warmup_iters (int): The number of iterations or epochs that warmup
            lasts. Note when by_epoch == True, warmup_iters means the number 
            of epochs that warmup lasts, otherwise means the number of 
            iteration that warmup lasts
        @warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        @runner (dict): Configs for running. Run by epoches or iters.
    """

    def __init__(self,
                 by_epoch: bool=True,
                 warmup: str=None,
                 warmup_iters: int=0,
                 warmup_ratio: float=0.1,
                 runner: dict={}):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'
        
        if runner is None:
            raise RuntimeError('runner should be set.')

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.runner = runner

        self.max_iters = None
        self.max_epoches = None
        if 'IterBasedRunner' in self.runner.type:
            self.max_iters = self.runner.max_iters
            assert self.by_epoch==False
            self.warmup_by_epoch = False
        elif 'EpochBasedRunner' in self.runner.type:
            self.max_epoches = self.runner.max_epoches
            assert self.by_epoch==True
            self.warmup_by_epoch = True
        else:
            raise ValueError(f'{self.runner.type} is not a supported type for running.')
        
        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed
        self._step_count = 0

    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups,
                                       lr_groups):
                param_group['lr'] = lr

    def get_lr(self, _iter, max_iter, base_lr):
        raise NotImplementedError

    def get_regular_lr(self, _iter, optimizer):
        max_iters = self.max_iters if not self.by_epoch else self.max_epoches

        if isinstance(optimizer, dict):
            lr_groups = {}
            for k in optimizer.keys():
                _lr_group = [
                    self.get_lr(_iter, max_iters,  _base_lr)
                    for _base_lr in self.base_lr[k]
                ]
                lr_groups.update({k: _lr_group})

            return lr_groups
        else:
            return [self.get_lr(_iter, max_iters, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):

        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 -
                                                           self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def before_run(self, optimizer):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if isinstance(optimizer, dict):
            self.base_lr = {}
            for k, optim in optimizer.items():
                for group in optim.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                _base_lr = [
                    group['initial_lr'] for group in optim.param_groups
                ]
                self.base_lr.update({k: _base_lr})
        else:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            self.base_lr = [
                group['initial_lr'] for group in optimizer.param_groups
            ]

    def after_train_epoch(self, optimizer):
        self._step_count += 1
        curr_epoch = self._step_count
        self.regular_lr = self.get_regular_lr(curr_epoch, optimizer)
        if self.warmup is None or curr_epoch > self.warmup_epoches:
            self._set_lr(optimizer, self.regular_lr)
        else:
            #self.warmup_iters = int(self.warmup_epochs * epoch_len)
            warmup_lr = self.get_warmup_lr(curr_epoch)
            self._set_lr(optimizer, warmup_lr)

    def after_train_iter(self, optimizer):
        self._step_count += 1
        cur_iter = self._step_count
        self.regular_lr = self.get_regular_lr(cur_iter, optimizer)
        if self.warmup is None or cur_iter >= self.warmup_iters:
            self._set_lr(optimizer, self.regular_lr)
        else:
            warmup_lr = self.get_warmup_lr(cur_iter)
            self._set_lr(optimizer, warmup_lr)

    def get_curr_lr(self, cur_iter):
        if self.warmup is None or cur_iter >= self.warmup_iters:
            return self.regular_lr
        else:
            return self.get_warmup_lr(cur_iter)
    
    def state_dict(self):
        """
        Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            @state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)


class PolyLrUpdater(LrUpdater):

    def __init__(self, power=1., min_lr=0., **kwargs):
        self.power = power
        self.min_lr = min_lr
        super(PolyLrUpdater, self).__init__(**kwargs)

    def get_lr(self, _iter, max_iters, base_lr):
        progress = _iter
        max_progress = max_iters
        coeff = (1 - progress / max_progress)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr


def build_lr_schedule_with_cfg(cfg):
    # build learning rate schedule with config.
    lr_config = copy.deepcopy(cfg.lr_config)
    policy = lr_config.pop('policy')
    if cfg.lr_config.policy == 'poly':
        schedule = PolyLrUpdater(runner=cfg.runner, **lr_config)
    else:
        raise RuntimeError(f'{cfg.lr_config.policy} \
                            is not supported in this framework.')
    return schedule
  

#def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
#    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
#    lr = base_lr * (multiplier ** (epoch // step_epoch))
#    return lr

def register_torch_optimizers():
    torch_optimizers = {}
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            torch_optimizers[module_name] = _optim
    return torch_optimizers


TORCH_OPTIMIZER = register_torch_optimizers()

def build_optimizer_with_cfg(cfg, model):
    # encoder_parameters = []
    # decoder_parameters = []
    # nongrad_parameters = []
    # for key, value in dict(model.named_parameters()).items():
    #     if value.requires_grad:
    #         if 'encoder' in key:
    #             encoder_parameters.append(value)
    #         else:
    #             decoder_parameters.append(value)
    #     else:
    #         nongrad_parameters.append(value)

    #params = [{"params": filter(lambda p: p.requires_grad, model.parameters())}]
    optim_cfg = copy.deepcopy(cfg.optimizer)
    optim_type = optim_cfg.pop('type', None)
    
    if optim_type is None:
        raise RuntimeError(f'{optim_type} is not set')
    if optim_type not in TORCH_OPTIMIZER:
        raise RuntimeError(f'{optim_type} is not supported in torch {torch.__version__}')
    if 'others' not in optim_cfg:
        optim_cfg['others'] = optim_cfg['decoder']

    def match(key1, key_list, strict_match=False):
        if not strict_match:
            for k in key_list:
                if k in key1:
                    return k
        else:
            for k in key_list:
                if k == key1.split('.')[1]:
                    return k        
        return None
    optim_obj = TORCH_OPTIMIZER[optim_type]
    matching_type = optim_cfg.pop('strict_match', False)

    module_names = optim_cfg.keys()
    model_parameters = {i: [] for i in module_names}
    model_parameters['others'] = []
    nongrad_parameters = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            match_key =  match(key, module_names, matching_type)
            # if optim_cfg[match_key]['lr'] == 0:
            #     value.requires_grad=False
            #     continue
            if match_key is None:
                model_parameters['others'].append(value)
            else:
                model_parameters[match_key].append(value)
        else:
            nongrad_parameters.append(value)

    optims = [{'params':model_parameters[k], **optim_cfg[k]} for k in optim_cfg.keys()]
    optimizer = optim_obj(optims)
    # optim_args_encoder = optim_cfg.optimizer.encoder
    # optim_args_decoder = optim_cfg.optimizer.decoder
    # optimizer = optim_obj(
    #     [{'params':encoder_parameters, **optim_args_encoder},
    #     {'params':decoder_parameters, **optim_args_decoder},
    # ])

    return optimizer


def load_ckpt(load_path, model, optimizer=None, scheduler=None, strict_match=True, loss_scaler=None): 
    """
        Load the check point for resuming training or finetuning.
    """
    logger = logging.getLogger()
    if os.path.isfile(load_path):
        if main_process():
            logger.info(f"Loading weight '{load_path}'")
        checkpoint = torch.load(load_path, map_location="cpu")
        ckpt_state_dict = checkpoint['model_state_dict']
        model.module.load_state_dict(ckpt_state_dict, strict=strict_match)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if loss_scaler is not None and 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
            print('Loss scaler loaded', loss_scaler)
        del ckpt_state_dict
        del checkpoint
        if main_process():
            logger.info(f"Successfully loaded weight: '{load_path}'")
            if scheduler is not None and optimizer is not None:
                logger.info(f"Resume training from: '{load_path}'")
    else:
        if main_process():
            raise RuntimeError(f"No weight found at '{load_path}'")
    return model, optimizer, scheduler, loss_scaler


def save_ckpt(cfg, model, optimizer, scheduler, curr_iter=0, curr_epoch=None, loss_scaler=None):
    """
        Save the model, optimizer, lr scheduler.
    """
    logger = logging.getLogger()

    if 'IterBasedRunner' in cfg.runner.type:
        max_iters = cfg.runner.max_iters
    elif 'EpochBasedRunner' in cfg.runner.type:
        max_iters = cfg.runner.max_epoches    
    else:
        raise TypeError(f'{cfg.runner.type} is not supported')

    ckpt = dict(model_state_dict=model.module.state_dict(),
                optimizer=optimizer.state_dict(),
                max_iter=cfg.runner.max_iters if 'max_iters' in cfg.runner \
                         else cfg.runner.max_epoches,
                scheduler=scheduler.state_dict(),
                # current_iter=curr_iter,
                # current_epoch=curr_epoch,
                )
    if loss_scaler is not None:
        # amp state_dict
        ckpt.update(dict(scaler=loss_scaler.state_dict()))

    ckpt_dir = os.path.join(cfg.work_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    save_name = os.path.join(ckpt_dir, 'step%08d.pth' % curr_iter)
    saved_ckpts = glob.glob(ckpt_dir + '/step*.pth')
    torch.save(ckpt, save_name)

    # keep the last 8 ckpts
    if len(saved_ckpts) > 8:
        saved_ckpts.sort()
        os.remove(saved_ckpts.pop(0))

    logger.info(f'Save model: {save_name}')



if __name__ == '__main__':
    print(TORCH_OPTIMIZER)