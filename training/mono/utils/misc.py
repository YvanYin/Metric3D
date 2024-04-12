


import os
import torch
try:
    from torch._six import inf
except:
    from torch import inf


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        #self._scaler = torch.cuda.amp.GradScaler(init_scale=16384) #init_scale=4096.0
        self._scaler = torch.cuda.amp.GradScaler(init_scale=1)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                try:
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad, error_if_nonfinite=True)
                except:
                    print('NAN gradient ....')
            else:
                raise NotImplementedError
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return True
        #return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def is_bf16_supported():
    """Returns a bool indicating if the current CUDA device supports dtype bfloat16"""
    cu_vers = torch.version.cuda
    if cu_vers is not None:
        cuda_maj_decide = int(cu_vers.split('.')[0]) >= 11
    else:
        cuda_maj_decide = False
    return torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8 and cuda_maj_decide