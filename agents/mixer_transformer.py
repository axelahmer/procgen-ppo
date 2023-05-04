from agents.impala import ConvSequence
from torch.distributions import Categorical
import torch
import torch.nn as nn


class TransformerMixer(nn.Module):
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

        # Hyperparameters for each stream
        # Common hyperparameters
        kernel_size = 4
        stride = 1
        padding = 0

        # Configuration stream
        conf_emb_size = 512

        # Attention stream
        attn_emb_size = 64
        attn_num_layers = 2
        attn_num_heads = 4
        attn_ff_dim = attn_emb_size * 4
        attn_dropout = 0
        attn_bottleneck = 32

        # Attention stream with positional encoding
        attn_pos_emb_size = 64
        attn_pos_num_layers = 4
        attn_pos_num_heads = 8
        attn_pos_ff_dim = attn_pos_emb_size * 4
        attn_pos_dropout = 0
        attn_pos_bottleneck = 16
        ##################

        total_emb_size = conf_emb_size + attn_pos_bottleneck

        self.conf_embedding = nn.Conv2d(shape[0], conf_emb_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.attn_embedding = nn.Conv2d(shape[0], attn_emb_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.attn_pos_embedding = nn.Conv2d(shape[0], attn_pos_emb_size, kernel_size=kernel_size, stride=stride, padding=padding)

        num_patches = ((shape[1] - kernel_size + 2 * padding) // stride + 1) * ((shape[2] - kernel_size + 2 * padding) // stride + 1)
        self.position_embedding = nn.Embedding(num_patches, attn_pos_emb_size)

        attn_encoder_layer = nn.TransformerEncoderLayer(d_model=attn_emb_size, nhead=attn_num_heads, batch_first=True, dim_feedforward=attn_ff_dim, dropout=attn_dropout, norm_first=True)
        self.attn_encoder = nn.TransformerEncoder(attn_encoder_layer, num_layers=attn_num_layers)
        self.attn_bottle = nn.Linear(attn_emb_size, attn_bottleneck)

        attn_pos_encoder_layer = nn.TransformerEncoderLayer(d_model=attn_pos_emb_size, nhead=attn_pos_num_heads, batch_first=True, dim_feedforward=attn_pos_ff_dim, dropout=attn_pos_dropout, norm_first=True)
        self.attn_pos_encoder = nn.TransformerEncoder(attn_pos_encoder_layer, num_layers=attn_pos_num_layers)
        self.attn_pos_bottle = nn.Linear(attn_pos_emb_size, attn_pos_bottleneck)

        self.policy_head = nn.Linear(total_emb_size, self.num_outputs)
        self.policy_weights = nn.Linear(total_emb_size, 1)

        self.value_head = nn.Linear(total_emb_size, 1)
        self.value_weights = nn.Linear(total_emb_size, 1)

        self.norm = nn.LayerNorm(total_emb_size)

        self.register_buffer("position_ids", torch.arange(num_patches).expand((1, -1)))


    def get_value(self, x):
        _, _, _, v = self.get_action_and_value(x)
        return v

    def get_action_and_value(self, x, action=None):
        x = self.conv_seqs(x.permute((0, 3, 1, 2)) / 255.0)
        x = nn.functional.relu(x)

        # conf stream (B x P x conf_emb_size)
        conf_emb = self.conf_embedding(x)
        conf_emb = conf_emb.flatten(2).permute(0, 2, 1)

        # attn stream (B x P x attn_emb_size)
        attn_emb = self.attn_embedding(x)
        attn_emb = attn_emb.flatten(2).permute(0, 2, 1)
        attn_emb = self.attn_encoder(attn_emb)
        attn_emb = self.attn_bottle(attn_emb)

        # attn_pos stream (B x P x attn_pos_emb_size)
        attn_pos_emb = self.attn_pos_embedding(x)
        attn_pos_emb = attn_pos_emb.flatten(2).permute(0, 2, 1)
        attn_pos_emb = attn_pos_emb + self.position_embedding(self.position_ids)
        attn_pos_emb = self.attn_pos_encoder(attn_pos_emb)
        attn_pos_emb = self.attn_pos_bottle(attn_pos_emb)

        # Concatenate streams (B x P x total_emb_size)
        x = torch.cat((conf_emb, attn_pos_emb), dim=-1)
        x = nn.functional.relu(x)
        # x = self.norm(x)
        
        

        logits = (self.policy_head(x) * torch.softmax(self.policy_weights(x), dim=1)).sum(dim=1)
        value = (self.value_head(x) * torch.softmax(self.value_weights(x), dim=1)).sum(dim=1)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value


