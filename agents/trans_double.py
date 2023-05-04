from agents.impala import ConvSequence
from torch.distributions import Categorical
import torch
import torch.nn as nn


class DoubleTransformer(nn.Module):
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
        attn_emb_size = 64
        base_emb_size = 512
        num_layers = 1
        num_heads = 2
        ff_dim = attn_emb_size * 2
        dropout = 0
        shared_linear_size = 256
        ##################

        self.base_embedding = nn.Conv2d(shape[0], base_emb_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.attn_embedding = nn.Conv2d(shape[0], attn_emb_size, kernel_size=kernel_size, stride=stride, padding=padding)

        encoder_layer = nn.TransformerEncoderLayer(d_model=attn_emb_size, nhead=num_heads, batch_first=True, dim_feedforward=ff_dim, dropout=dropout, norm_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        num_patches = ((shape[1] - kernel_size + 2 * padding) // stride + 1) * ((shape[2] - kernel_size + 2 * padding) // stride + 1)
        self.position_embedding = nn.Embedding(num_patches, attn_emb_size)

        combined_emb_size = attn_emb_size + base_emb_size

        self.shared_linear = nn.Linear(combined_emb_size, shared_linear_size)

        self.policy_head = nn.Linear(shared_linear_size, self.num_outputs)
        self.policy_weights = nn.Linear(shared_linear_size, 1)

        self.value_head = nn.Linear(shared_linear_size, 1)
        self.value_weights = nn.Linear(shared_linear_size, 1)

        self.norm = nn.LayerNorm(base_emb_size)

        self.register_buffer("position_ids", torch.arange(num_patches).expand((1, -1)))

    def get_value(self, x):
        _, _, _, v = self.get_action_and_value(x)
        return v

    def get_action_and_value(self, x, action=None):
        x = self.conv_seqs(x.permute((0, 3, 1, 2)) / 255.0)
        x = nn.functional.relu(x)

        # Base stream (B x P x base_emb_size)
        x_base = self.base_embedding(x)
        x_base = x_base.flatten(2).permute(0, 2, 1)

        # Attn stream (B x P x attn_emb_size)
        x_attn = self.attn_embedding(x)
        x_attn = x_attn.flatten(2).permute(0, 2, 1)
        x_attn = x_attn + self.position_embedding(self.position_ids)
        x_attn = self.encoder(x_attn)

        # Concatenate streams (B x P x combined_emb_size)
        x_combined = torch.cat((self.norm(x_base), x_attn), dim=2)
        x_combined = nn.functional.relu(x_combined)

        # Shared linear (B x P x shared_linear_size)
        x = self.shared_linear(x_combined)
        x = nn.functional.relu(x)

        logits = (self.policy_head(x) * torch.softmax(self.policy_weights(x), dim=1)).sum(dim=1)
        value = (self.value_head(x) * torch.softmax(self.value_weights(x), dim=1)).sum(dim=1)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value