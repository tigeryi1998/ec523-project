import torch
import torch.nn as nn
import copy

class EMA:
    
    def __init__(self, model: nn.Module, decay_rate, device=None):
        # copy the model, this model is the model that is in "target" mode
        self.model = copy.deepcopy(model)
        # model should not be changed by calls to backward()
        self.model.requires_grad_(False)
        # update model if training on CUDA
        self.device = device
        self.model.to(device)
        # set decay rate
        self.decay = decay_rate
        # the number of times we have stepped the EMA, needed to eventually taper down delta
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
        
        
    @staticmethod
    def calc_annealed_rate(start, end, curr_step, total_steps):
        # this function calculates the next step for the decay rate
        r = end - start
        pct_remaining = 1 - curr_step / total_steps
        return end - r * pct_remaining