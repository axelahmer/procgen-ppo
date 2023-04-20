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
    


class MixerTransformerAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.num_outputs = envs.single_action_space.n
        h, w, c = envs.single_observation_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.Sequential(*conv_seqs)

        emb_size = 64
        lat_size = 896
        comb_size = emb_size + lat_size

        self.latent = nn.Conv2d(32, lat_size, stride=1, kernel_size=4)
        self.emb = nn.Conv2d(32, emb_size, stride=1, kernel_size=4)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=2, batch_first=True, dim_feedforward=emb_size * 2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.policy_head = nn.Linear(comb_size, self.num_outputs)
        self.policy_weights = nn.Linear(comb_size, 1)

        self.value_head = nn.Linear(comb_size, 1)
        self.value_weights = nn.Linear(comb_size, 1)

        self.lat_norm = nn.LayerNorm(lat_size)  # Add LayerNorm for lat

        # Precompute positional encoding
        seq_len = (shape[1] - 3) * (shape[2] - 3)
        self.register_buffer("precomputed_positional_encoding", self.positional_encoding(seq_len, emb_size))

    def add_positional_encoding(self, x):
        x = x + self.precomputed_positional_encoding.unsqueeze(0)
        return x

    def positional_encoding(self, seq_len, d_model):
        pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe

    def get_value(self, x):
        _, _, _, v = self.get_action_and_value(x)
        return v

    def get_action_and_value(self, x, action=None):
        x = self.conv_seqs(x.permute((0, 3, 1, 2)) / 255.0)
        x = nn.functional.relu(x)

        attn = self.emb(x).flatten(2).permute(0, 2, 1)
        attn = self.add_positional_encoding(attn)
        attn = self.encoder(attn)

        lat = self.latent(x).flatten(2).permute(0, 2, 1)
        lat = self.lat_norm(lat)  # Apply LayerNorm to lat

        x = torch.cat((lat, attn), dim=2)
        x = nn.functional.relu(x)

        logits = (self.policy_head(x) * torch.softmax(self.policy_weights(x), dim=1)).sum(dim=1)
        value = (self.value_head(x) * torch.softmax(self.value_weights(x), dim=1)).sum(dim=1)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value