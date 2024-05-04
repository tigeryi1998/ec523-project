import os
import copy

import torch
import torch.nn as nn


class EMA:
    """
    Modified version of class fairseq.models.ema.EMAModule.

    Args:
        model (nn.Module):
        cfg (DictConfig):
        device (str):
        skip_keys (list): The keys to skip assigning averaged weights to.
    """

    def __init__(self, model: nn.Module, cfg, skip_keys=None):
        self.model = self.deepcopy_model(model)
        self.model.requires_grad_(False)
        self.cfg = cfg
        self.device = cfg.device
        self.model.to(self.device)
        self.decay = self.cfg.model.ema_decay
        self.update_count = 0

    def step(self, n_model: nn.Module):
        """
        A single step of the EMA parameterization of the teacher's parameters

        Args:
            n_model (nn.Module): this is the student model, that the teacher will be following with EMA
        """
        # current parameters
        ema_state_dict = {}
        ema_params = self.model.state_dict()
        
        # update parameters as a the exponential moving average of n_model
        for key, param in n_model.state_dict().items():
            xx = ema_params[key].float()
            xx.mul_(self.decay_rate)
            xx = xx.add(param.to(dtype=xx.dtype).mul(1-self.decay))
            ema_state_dict[key] = xx
        
        # load the updated parameters back into model
        self.model.load_state_dict(ema_state_dict, strict=False)
        # update count
        self.update_count += 1
        
        
    def set_decay(self, decay_rate):
        self.decay_rate = decay_rate
