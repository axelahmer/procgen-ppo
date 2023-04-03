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
    



# TODO NOT ACTUALLY IMPLEMENTED YET

class MixerAgentFlatVal(nn.Module):
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
            nn.Conv2d(in_channels=32, out_channels=1024, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=self.num_outputs + 3, kernel_size=1, stride=1),
        )

    def get_value(self, x):
        _, _, _, v = self.get_action_and_value(x)
        return v
    
    def get_action_and_value(self, x, action=None):
        
        # embed state
        x = self.conv_seqs(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        x = self.experts(x)

        # expert forward pass
        x = x.flatten(2)
        logits = x.narrow(1, 0, self.num_outputs)  # N x A X self.num_actors
        value = x.narrow(1, self.num_outputs, 1)  # N x 1 X self.num_actors
        weights_logits = x.narrow(1, self.num_outputs + 1, 1)  # N x 1 X self.num_actors
        weights_value = x.narrow(1, self.num_outputs + 2, 1)  # N x 1 X self.num_actors

        # normalize weights
        weights_logits = nn.functional.softmax(weights_logits, dim=2)
        weights_value = nn.functional.softmax(weights_value, dim=2)

        # weighted sum
        logits = logits.mul(weights_logits).sum(2)
        value = value.mul(weights_value).sum(2)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value