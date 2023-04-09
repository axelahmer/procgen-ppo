import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)
    



class MixerAgentSigmoidIndividualEntropy(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.num_outputs = envs.single_action_space.n
        h, w, c = envs.single_observation_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16,32,32]:  # regular: [16, 32, 32]
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.Sequential(*conv_seqs)

        self.experts = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=768, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=768, out_channels=self.num_outputs + 1, kernel_size=1, stride=1),
        )
        
        self.hidden_fc = nn.Linear(in_features=8*8*32, out_features=768)
        self.value_fc = nn.Linear(in_features=768, out_features=1)


    def get_value(self, x):
        _, _, _, v, _ = self.get_action_and_value(x)
        return v
    
    def get_action_and_value(self, x, action=None):
        
        # embed state
        x = self.conv_seqs(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        
        x_flat = x.flatten(1)
        x_flat = self.hidden_fc(x_flat)
        x_flat = nn.functional.relu(x_flat)
        value = self.value_fc(x_flat)
        
        # expert forward pass
        x = self.experts(x)
        x = x.flatten(2)
        logits = x.narrow(1, 0, self.num_outputs)  # N x A X self.num_actors
        weights_logits = x.narrow(1, self.num_outputs, 1)  # N x 1 X self.num_actors
        weights_logits = torch.sigmoid(weights_logits).mul(0.5) # If multiplied by 2, the default weighting is 1. But think we should divide by sqrt(actors), or something just under. Kinda guessing that this might be ok.

        # weighted sum
        final_logits = logits.mul(weights_logits).sum(2)

        probs = Categorical(logits=final_logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value, logits
