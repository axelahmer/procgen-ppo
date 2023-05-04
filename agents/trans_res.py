from agents.impala import ConvSequence
from torch.distributions import Categorical
import torch
import torch.nn as nn


class ResidualAttention(nn.Module):
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

        # Hyperparameters
        kernel_size = 4
        stride = 1
        padding = 0
        emb_size = 256
        num_layers = 1
        num_heads = 4
        ff_dim = 32
        dropout = 0
        linear_size = 512
        ##################

        self.embedding = nn.Conv2d(shape[0], emb_size, kernel_size=kernel_size, stride=stride, padding=padding)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, batch_first=True, dim_feedforward=ff_dim, dropout=dropout, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        num_patches = ((shape[1] - kernel_size + 2 * padding) // stride + 1) * ((shape[2] - kernel_size + 2 * padding) // stride + 1)
        self.position_embedding = nn.Embedding(num_patches, emb_size)

        self.linear = nn.Linear(emb_size, linear_size)

        self.policy_head = nn.Linear(linear_size, self.num_outputs)
        self.policy_weights = nn.Linear(linear_size, 1)

        self.value_head = nn.Linear(linear_size, 1)
        self.value_weights = nn.Linear(linear_size, 1)

        self.register_buffer("position_ids", torch.arange(num_patches).expand((1, -1)))

    def get_value(self, x):
        _, _, _, v = self.get_action_and_value(x)
        return v

    def get_action_and_value(self, x, action=None):
        x = self.conv_seqs(x.permute((0, 3, 1, 2)) / 255.0)
        x = nn.functional.relu(x)

        x = self.embedding(x).flatten(2).permute(0, 2, 1) # B x P x emb_size
        attn = x + self.position_embedding(self.position_ids)
        attn = self.encoder(attn)

        x = x + attn
        x = nn.functional.relu(x)

        x = self.linear(x)
        x = nn.functional.relu(x)

        logits = (self.policy_head(x) * torch.softmax(self.policy_weights(x), dim=1)).sum(dim=1)
        value = (self.value_head(x) * torch.softmax(self.value_weights(x), dim=1)).sum(dim=1)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value