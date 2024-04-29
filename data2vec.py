import torch
import torch.nn as nn
import torch.nn.functional as F
from ema import EMA



class Data2Vec(nn.Module):
    
    def __init__(self, encoder, modality, embed_dim, ema_decay, ema_end_decay, ema_anneal_end_step, device):
        self.encoder = encoder
        self.modality = modality
        self.embed_dim = embed_dim
        self.EMAModule = EMA(self.encoder, decay_rate=ema_decay, device=device)
        self.device = device
        
        # building the regression head
        if self.modality == 'text':
            self.regression_head = None
        elif self.modality == 'image':
            self.regression_head = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            raise Exception('Given modality is not accepted: ', str(modality))
            
        # parameters for the EMA module
        self.ema_decay = ema_decay                          # starting decay rate
        self.ema_end_decay = ema_end_decay                  # ending decay rate
        self.ema_anneal_end_step = ema_anneal_end_step      # how many steps it should take to reach decay rate (at max)
        
        
    def stepEMA(self):
        '''
        Performs a step in the EMA module (as the teacher model)
        '''
        # find new decay rate if necessary
        if self.ema_decay >= self.ema_end_decay:
            if self.EMAModule.update_count >= self.ema_anneal_end_step:
                # get decay as the ending decay
                decay = self.ema_end_decay
            else:
                # get decay based on update count
                delta = self.ema_end_decay - self.ema_decay
                updates_remaining = 1 - (self.EMAModule.update_count / self.ema_anneal_end_step)
                decay = self.ema_end_decay - (delta * updates_remaining) 
            
            # if we changed, set new decay rate in EMA module
            self.EMAModule.set_decay(decay)
            
        # perform step in EMA
        if self.EMAModule.decay < 1:
            self.EMAModule.step()
        
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
        y = y[mask]                                                                 # 7) Apply mask to y
        
        return x, y # this output is only for teacher, x and y represent the 
            
            