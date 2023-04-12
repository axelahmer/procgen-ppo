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

        self.experts_1 = nn.Conv2d(in_channels=32, out_channels=504, kernel_size=4, stride=1)
        self.experts_2 = nn.Conv2d(in_channels=504, out_channels=self.num_outputs+1, kernel_size=1, stride=1)
        
        self.hidden_fc = nn.Linear(in_features=8*8*32, out_features=126)
        self.value_fc = nn.Linear(in_features=126, out_features=1)
        
        self.count_parameters()
        
        self_dict = self.state_dict()
        conv_weights = self_dict['experts_2.weight']
        conv_bias = self_dict['experts_2.bias']
        
        new_weight_expert_2a = self.experts_2.weight.mul(0.01)
        new_bias_expert_2a = self.experts_2.bias.mul(0.01)
        
        self_dict['experts_2.weight'] = new_weight_expert_2a
        self_dict['experts_2.bias'] = new_bias_expert_2a
        
        self.load_state_dict(self_dict)

    def get_value(self, x):
        _, _, _, v = self.get_action_and_value(x)
        return v
    
    def get_action_and_value(self, x, action=None):
        
        # embed state
        x = self.conv_seqs(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        
        x_flat = nn.functional.relu(x)
        x_flat = x_flat.flatten(1)
        x_flat = self.hidden_fc(x_flat)
        x_flat = nn.functional.relu(x_flat)
        value = self.value_fc(x_flat)
        
        # expert forward pass
        x = nn.functional.relu(x)
        x = self.experts_1(x)
        x = nn.functional.relu(x)
        x = self.experts_2(x)
        x = x.flatten(2)
        logits = x.narrow(1, 0, self.num_outputs)  # N x A X self.num_actors
        weights_logits = x.narrow(1, self.num_outputs, 1)  # N x 1 X self.num_actors
        weights_logits = torch.sigmoid(weights_logits).mul(2.0) #.mul(0.5) # TODO: Figure out if any adjustment factor is needed here.

        # weighted sum
        logits_weights_detached = logits.mul(weights_logits.detach()).sum(2)
        logits = logits.mul(weights_logits).sum(2)
        
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

