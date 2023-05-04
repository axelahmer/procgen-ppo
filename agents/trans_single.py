from agents.impala import ConvSequence
from torch.distributions import Categorical
import torch
import torch.nn as nn


class TransformerSingle(nn.Module):
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

        # Hyperparameters for the single stream
        kernel_size = 4
        stride = 1
        padding = 0
        emb_size = 256
        num_layers = 1
        num_heads = 16
        ff_dim = emb_size * 1
        dropout = 0
        ##################

        self.single_stream_embedding = nn.Conv2d(shape[0], emb_size, kernel_size=kernel_size, stride=stride, padding=padding)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, batch_first=True, dim_feedforward=ff_dim, dropout=dropout, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_head = nn.Linear(emb_size, self.num_outputs)
        self.policy_weights = nn.Linear(emb_size, 1)

        self.value_head = nn.Linear(emb_size, 1)
        self.value_weights = nn.Linear(emb_size, 1)

    def get_value(self, x):
        _, _, _, v = self.get_action_and_value(x)
        return v

    def get_action_and_value(self, x, action=None):
        x = self.conv_seqs(x.permute((0, 3, 1, 2)) / 255.0)
        x = nn.functional.relu(x)

        # Single stream (B x P x emb_size)
        x = self.single_stream_embedding(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.encoder(x)
        x = nn.functional.relu(x)

        logits = (self.policy_head(x) * torch.softmax(self.policy_weights(x), dim=1)).sum(dim=1)
        value = (self.value_head(x) * torch.softmax(self.value_weights(x), dim=1)).sum(dim=1)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value
