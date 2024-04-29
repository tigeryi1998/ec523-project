import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class Encoder(nn.Module):
    """
    Encoder model using HuggingFace Transformers for vision e.g, BeiT

    Args:
        cfg: An omegaconf.DictConf instance containing all the configurations.
        **kwargs: extra args which are set as dataset properties
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.checkpoint = None
        model_config = AutoConfig.from_pretrained(checkpoint)
        self.encoder = AutoModel.from_config(model_config)
        self.vocab_size = None
        self.mask_token = None

    def forward(self, inputs, mask=None, **kwargs):
        """
        Forward inputs through the encoder and extract transformer/attention layers outputs

        Args:
            inputs: input pixels with shape [batch_size, channels, height, width]
            mask: bool masked indices
            **kwargs: keyword args specific to the encoder's forward method

        Returns:
            A dictionary of the encoder outputs including transformer layers outputs and attentions outputs
        """
        # Note: inputs are already masked for MIM so mask is not used
        outputs = self.encoder(pixel_values=inputs, output_hidden_states=True, output_attentions=True, **kwargs)
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