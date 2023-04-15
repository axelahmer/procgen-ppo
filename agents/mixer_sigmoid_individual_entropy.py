import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from prettytable import PrettyTable
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

        self.experts_pol = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=312, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=312, out_channels=self.num_outputs + 1, kernel_size=1, stride=1),
        )
        
        self.experts_val = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=312, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=312, out_channels=2, kernel_size=1, stride=1),
        )
        self.count_parameters()

    def get_value(self, x):
        _, _, _, v = self.get_action_and_value(x)
        return v
    
    def get_action_and_value(self, x, action=None):
        
        # embed state
        x = self.conv_seqs(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        
        x_val = self.experts_val(x)
        x_val = x_val.flatten(2)
        value = x_val.narrow(1, 0, 1)  # N x 1 X self.num_actors
        weights_value = x_val.narrow(1, 1, 1)  # N x 1 X self.num_actors
        
        # expert forward pass
        x = self.experts_pol(x)
        x = x.flatten(2)
        logits = x.narrow(1, 0, self.num_outputs)  # N x A X self.num_actors
        weights_logits = x.narrow(1, self.num_outputs, 1)  # N x 1 X self.num_actors

        weights_logits = nn.functional.softmax(weights_logits, dim=2)
        weights_value = nn.functional.softmax(weights_value, dim=2)

        # weighted sum
        logits_weights_detached = logits.mul(weights_logits.detach()).sum(2)
        logits = logits.mul(weights_logits).sum(2)
        value = value.mul(weights_value).sum(2)
        
        probs_weights_detached = Categorical(logits=logits_weights_detached)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs_weights_detached.entropy(), value
        
    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
        

#+-------------------------------------+------------+
#|               Modules               | Parameters |
#+-------------------------------------+------------+
#|       conv_seqs.0.conv.weight       |    432     |
#|        conv_seqs.0.conv.bias        |     16     |
#| conv_seqs.0.res_block0.conv0.weight |    2304    |
#|  conv_seqs.0.res_block0.conv0.bias  |     16     |
#| conv_seqs.0.res_block0.conv1.weight |    2304    |
#|  conv_seqs.0.res_block0.conv1.bias  |     16     |
#| conv_seqs.0.res_block1.conv0.weight |    2304    |
#|  conv_seqs.0.res_block1.conv0.bias  |     16     |
#| conv_seqs.0.res_block1.conv1.weight |    2304    |
#|  conv_seqs.0.res_block1.conv1.bias  |     16     |
#|       conv_seqs.1.conv.weight       |    4608    |
#|        conv_seqs.1.conv.bias        |     32     |
#| conv_seqs.1.res_block0.conv0.weight |    9216    |
#|  conv_seqs.1.res_block0.conv0.bias  |     32     |
#| conv_seqs.1.res_block0.conv1.weight |    9216    |
#|  conv_seqs.1.res_block0.conv1.bias  |     32     |
#| conv_seqs.1.res_block1.conv0.weight |    9216    |
#|  conv_seqs.1.res_block1.conv0.bias  |     32     |
#| conv_seqs.1.res_block1.conv1.weight |    9216    |
#|  conv_seqs.1.res_block1.conv1.bias  |     32     |
#|       conv_seqs.2.conv.weight       |    9216    |
#|        conv_seqs.2.conv.bias        |     32     |
#| conv_seqs.2.res_block0.conv0.weight |    9216    |
#|  conv_seqs.2.res_block0.conv0.bias  |     32     |
#| conv_seqs.2.res_block0.conv1.weight |    9216    |
#|  conv_seqs.2.res_block0.conv1.bias  |     32     |
#| conv_seqs.2.res_block1.conv0.weight |    9216    |
#|  conv_seqs.2.res_block1.conv0.bias  |     32     |
#| conv_seqs.2.res_block1.conv1.weight |    9216    |
#|  conv_seqs.2.res_block1.conv1.bias  |     32     |
#|           experts.1.weight          |   258048   |
#|            experts.1.bias           |    504     |
#|           experts.3.weight          |    8064    |
#|            experts.3.bias           |     16     |
#|           hidden_fc.weight          |   258048   |
#|            hidden_fc.bias           |    126     |
#|           value_fc.weight           |    126     |
#|            value_fc.bias            |     1      |
#+-------------------------------------+------------+
#Total Trainable Params: 622533


# If value part has 768 latent size:
#+-------------------------------------+------------+
#|               Modules               | Parameters |
#+-------------------------------------+------------+
#|       conv_seqs.0.conv.weight       |    432     |
#|        conv_seqs.0.conv.bias        |     16     |
#| conv_seqs.0.res_block0.conv0.weight |    2304    |
#|  conv_seqs.0.res_block0.conv0.bias  |     16     |
#| conv_seqs.0.res_block0.conv1.weight |    2304    |
#|  conv_seqs.0.res_block0.conv1.bias  |     16     |
#| conv_seqs.0.res_block1.conv0.weight |    2304    |
#|  conv_seqs.0.res_block1.conv0.bias  |     16     |
#| conv_seqs.0.res_block1.conv1.weight |    2304    |
#|  conv_seqs.0.res_block1.conv1.bias  |     16     |
#|       conv_seqs.1.conv.weight       |    4608    |
#|        conv_seqs.1.conv.bias        |     32     |
#| conv_seqs.1.res_block0.conv0.weight |    9216    |
#|  conv_seqs.1.res_block0.conv0.bias  |     32     |
#| conv_seqs.1.res_block0.conv1.weight |    9216    |
#|  conv_seqs.1.res_block0.conv1.bias  |     32     |
#| conv_seqs.1.res_block1.conv0.weight |    9216    |
#|  conv_seqs.1.res_block1.conv0.bias  |     32     |
#| conv_seqs.1.res_block1.conv1.weight |    9216    |
#|  conv_seqs.1.res_block1.conv1.bias  |     32     |
#|       conv_seqs.2.conv.weight       |    9216    |
#|        conv_seqs.2.conv.bias        |     32     |
#| conv_seqs.2.res_block0.conv0.weight |    9216    |
#|  conv_seqs.2.res_block0.conv0.bias  |     32     |
#| conv_seqs.2.res_block0.conv1.weight |    9216    |
#|  conv_seqs.2.res_block0.conv1.bias  |     32     |
#| conv_seqs.2.res_block1.conv0.weight |    9216    |
#|  conv_seqs.2.res_block1.conv0.bias  |     32     |
#| conv_seqs.2.res_block1.conv1.weight |    9216    |
#|  conv_seqs.2.res_block1.conv1.bias  |     32     |
#|           experts.1.weight          |   393216   |
#|            experts.1.bias           |    768     |
#|           experts.3.weight          |   12288    |
#|            experts.3.bias           |     16     |
#|           hidden_fc.weight          |  1572864   |
#|            hidden_fc.bias           |    768     |
#|           value_fc.weight           |    768     |
#|            value_fc.bias            |     1      |
#+-------------------------------------+------------+
#Total Trainable Params: 2078289

