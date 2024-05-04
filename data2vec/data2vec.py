import torch
import torch.nn as nn
import torch.nn.functional as F
from .ema import EMA


class Data2Vec(nn.Module):
    """
    Data2Vec main module.

    Args:
         encoder (nn.Module): The encoder module like BEiT, ViT, etc.
         cfg (omegaconf.DictConfig): The config containing model properties
    """
    MODALITIES = ['image', 'text']

    def __init__(self, encoder, cfg, **kwargs):
        super(Data2Vec, self).__init__()
        self.modality = cfg.modality
        self.embed_dim = cfg.model.embed_dim
        self.encoder = encoder
        self.__dict__.update(kwargs)

        self.cfg = cfg
        self.ema = EMA(self.encoder, cfg)  # EMA acts as the teacher
        

        self.cfg = cfg
        self.ema_decay = self.cfg.model.ema_decay
        self.ema_end_decay = self.cfg.model.ema_end_decay
        self.ema_anneal_end_step = self.cfg.model.ema_anneal_end_step

        if self.modality == 'text':
            self.regression_head = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 2),
				nn.GELU(),
				nn.Linear(self.embed_dim * 2, self.embed_dim)
        )

        if self.modality =='vision':
            self.regression_head =  nn.Linear(self.embed_dim, self.embed_dim)

    def stepEMA(self):
        '''
        Performs a step in the EMA module (as the teacher model)
        '''
        # find new decay rate if necessary
        if self.ema_decay >= self.ema_end_decay:
            if self.ema.update_count >= self.ema_anneal_end_step:
                # get decay as the ending decay
                decay = self.ema_end_decay
            else:
                # get decay based on update count
                delta = self.ema_end_decay - self.ema_decay
                updates_remaining = 1 - (self.ema.update_count / self.ema_anneal_end_step)
                decay = self.ema_end_decay - (delta * updates_remaining) 
            
            # if we changed, set new decay rate in EMA module
            self.ema.set_decay(decay)
            
        # perform step in EMA
        if self.ema.decay < 1:
            self.ema.step()

    def forward(self, src, k, target=None, mask=None):
        x = self.encoder(src, mask)['output']                                       # 1) pass source thru encoder (with mask), and get encoded rep
        
        # if we are in Student mode, we do not have a target, and we are simply building an encoded representation
        if target==None:
            return x
        
        # if we are in Teacher mode, we need to evualate syste
        with torch.no_grad():
            self.EMAModule.model.eval()
            y = self.EMAModule.model(target, ~mask)['states']                       # 2) Get transformer layers outputs 
            y = y[:k]                                                               # 3) Only keep top k layers outputs
            
            # normalizing layers
            y = [F.layer_norm(layer.float(), layer.shape[-1:]) for layer in y]      # 4) Normalize all outputs
            y = sum(y) / len(y)                                                     # 5) Get avg output across all top k layers
            
        x = self.regression_head(x[mask])                                           # 6) Regress x with mask (linear layer)
        y = y[mask]    
        
        return x, y
