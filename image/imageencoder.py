import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class Encoder(nn.Module):
    
    def __init__(self, checkpoint, vocab_size, mask_token):
        super(Encoder, self).__init__()
        
        self.vocab_sz = vocab_size
        self.mask_tkn = mask_token
        
        self.config = AutoConfig.from_pretrained(checkpoint)
        self.encoder = AutoModel.from_config(self.config)

    def forward(self, inputs):
        # Note: inputs are already masked for MIM so mask is not used
        outputs = self.encoder(pixel_values=inputs, output_hidden_states=True, output_attentions=True)
        encoder_states = outputs['hidden_states'][:-1]  # encoder layers outputs separately
        encoder_out = outputs['hidden_states'][-1]  # last encoder output (accumulated)
        attentions = outputs['attentions']

        # remove cls token from outputs
        encoder_states = [output[:, 1:, :] for output in encoder_states]
        encoder_out = encoder_out[:, 1:, :]
        attentions = [output[:, 1:, 1:] for output in attentions]

        return {
            'encoder_states': encoder_states,
            'encoder_out': encoder_out,
            'attentions': attentions
        }