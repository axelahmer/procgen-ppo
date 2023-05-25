from agents.impala import ConvSequence
from torch.distributions import Categorical
import torch
import torch.nn as nn

class TransformerVanilla(nn.Module):
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
        emb_size = 128
        kernel_size = 4
        stride = 1
        num_encoder_layers = 2
        num_heads = 2
        ff_dim = emb_size * 4
        dropout = 0
        ##################

        self.embedding = nn.Conv2d(shape[0], emb_size, kernel_size=kernel_size, stride=stride)
        # calc num_patches using kernel size and stride
        num_patches = ((shape[1] - kernel_size) // stride + 1) * ((shape[2] - kernel_size) // stride + 1)
        # print("num_patches", num_patches)
        self.position_embedding = nn.Embedding(num_patches, emb_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, batch_first=True, dim_feedforward=ff_dim, dropout=dropout, norm_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.policy_head = nn.Linear(emb_size*num_patches, self.num_outputs)
        self.value_head = nn.Linear(emb_size*num_patches, 1)

        self.register_buffer("position_ids", torch.arange(num_patches).expand((1, -1)))

    def get_value(self, x):
        _, _, _, v, _ = self.get_action_and_value(x)
        return v

    def get_action_and_value(self, x, action=None):

        x = self.conv_seqs(x.permute((0, 3, 1, 2)) / 255.0)
        x = nn.functional.relu(x)

        x = self.embedding(x).flatten(2).permute(0, 2, 1)
        x = x + self.position_embedding(self.position_ids)

        x = self.encoder(x)
        x = x.flatten(1)
        x = nn.functional.relu(x)

        logits = self.policy_head(x)
        value = self.value_head(x)

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value, logits
