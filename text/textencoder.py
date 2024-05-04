import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig

class TextEncoder(nn.Module):

    def __init__(self, checkpoint):
        super(TextEncoder, self).__init__()
        self.config = AutoConfig.from_pretrained(checkpoint)
        self.encoder = AutoModel.from_config(self.config)

    def forward(self, inputs, mask=None):
        # Note: inputs are already masked for MLM so mask is not used
        outputs = self.encoder(inputs, output_hidden_states=True, output_attentions=True)
        encoder_states = outputs['hidden_states'][:-1]  # encoder layers outputs separately
        encoder_out = outputs['hidden_states'][-1]      # last encoder output (accumulated)
        attentions = outputs['attentions']
        return {
            'encoder_states': encoder_states,
            'encoder_out': encoder_out,
            'attentions': attentions
        }