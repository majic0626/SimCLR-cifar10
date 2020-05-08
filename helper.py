import torch
import numpy as np
import cv2
from math import sin, pi


# def parallel(model):
#     num_gpu = torch.cuda.device_count()
#     model_name = model.__class__.__name__
#     if num_gpu > 1:  # >= 2
#         print("Your model {} trained in multiple gpus!".format(model_name))
#         gpu_ids = list(range(num_gpu))
#         model = nn.DataParallem(model, device_ids=gpu_ids)
#     else:  # == 1
#         if torch.cuda.is_available():
#             print("Your model {} trained in one gpu!".format(model_name))
#         else:  # == 0
#             print("Your model {} trained in cpu!".format(model_name))

#     return model


def visualize(loader, path, num=10, size=8):
    for ix, (sample1, sample2) in enumerate(loader):
        b, c, h, w = sample1["image"].size()
        if b < 64:
            raise ValueError("batch size < 64 cannot generate gallery for train data!!!")
        xi = sample1["image"].numpy()[:64].transpose(0, 2, c, 1).reshape(size, size, h, w, c)
        xj = sample2["image"].numpy()[:64].transpose(0, 2, c, 1).reshape(size, size, h, w, c)
        xi = np.swapaxes(xi, 1, 2).reshape(size * h, size * w, c)
        xj = np.swapaxes(xj, 1, 2).reshape(size * h, size * w, c)
        xi = (xi * 255).astype(np.uint8)
        xj = (xj * 255).astype(np.uint8)
        space = np.full((size * h, 20, c), 0, dtype=np.uint8)
        gallery = np.concatenate((xi, space, xj), axis=1)
        cv2.imwrite(path, gallery)
        if (ix + 1) == num:
            return None


def adjust_lr(opt, epoch, lr_init, T, warmup=10, lr_end=1e-5):
    if epoch < warmup:
        lr = lr_init * ((1 / warmup) * epoch)
    else:
        lr = lr_init - (lr_init - lr_end) * sin((pi / 2) * (epoch - warmup) / (T - warmup))
    for param_group in opt.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_linear_lr(opt, epoch, lr_init, T, lr_end=0):
    lr = lr_init - (lr_init - lr_end) * sin((pi / 2) * (epoch - 1) / (T - 1))
    for param_group in opt.param_groups:
        param_group['lr'] = lr
    return lr


# LARS (layer adapative scaling rate)
# please refer to "https://github.com/NVIDIA/apex/blob/d74fda260c403f775817470d87f810f816f3d615/apex/parallel/LARC.py"
class LARC(object):
    """
    :class:`LARC` is a pytorch implementation of both the scaling and clipping variants of LARC,
    in which the ratio between gradient and parameter magnitudes is used to calculate an adaptive
    local learning rate for each individual parameter. The algorithm is designed to improve
    convergence of large batch training.

    See https://arxiv.org/abs/1708.03888 for calculation of the local learning rate.
    In practice it modifies the gradients of parameters as a proxy for modifying the learning rate
    of the parameters. This design allows it to be used as a wrapper around any torch.optim Optimizer.
    ```
    model = ...
    optim = torch.optim.Adam(model.parameters(), lr=...)
    optim = LARC(optim)
    ```
    It can even be used in conjunction with apex.fp16_utils.FP16_optimizer.
    ```
    model = ...
    optim = torch.optim.Adam(model.parameters(), lr=...)
    optim = LARC(optim)
    optim = apex.fp16_utils.FP16_Optimizer(optim)
    ```
    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the lr. See https://arxiv.org/abs/1708.03888
        clip: Decides between clipping or scaling mode of LARC.
        If `clip=True` the learning rate is set to `min(optimizer_lr, local_lr)` for each parameter.
        If `clip=False` the learning rate is set to `local_lr*optimizer_lr`.
        eps: epsilon kludge to help with numerical stability while calculating adaptive_lr
    """

    def __init__(self, optimizer, trust_coefficient=0.02, clip=True, eps=1e-8):
        self.param_groups = optimizer.param_groups
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    def __repr__(self):
        return self.optim.__repr__()

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = self.trust_coefficient * (param_norm) / (grad_norm + param_norm * weight_decay + self.eps)

                        # clip learning rate for LARC
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr / group['lr'], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]
