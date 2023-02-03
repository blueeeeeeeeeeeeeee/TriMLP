import copy
import torch
import torch.nn as nn
from einops import rearrange
from TrainTools.train import train_multi


class PreNorm(nn.Module):
    def __init__(self, d_model, layer):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.layer = layer

    def forward(self, x):
        return x + self.layer(self.norm(x))


class FFN(nn.Module):
    def __init__(self, d_model, dropout):
        super(FFN, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_model * 4, bias=False)
        self.linear_2 = nn.Linear(d_model * 4, d_model, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.act(self.linear_1(x)))
        x = self.drop(self.linear_2(x))
        return x


class Block(nn.Module):
    def __init__(self, d_model, mixer_layer, ffn_layer, dropout):
        super(Block, self).__init__()
        self.mixer_layer = PreNorm(d_model, mixer_layer)
        self.ffn_layer = PreNorm(d_model, ffn_layer)
    
    def forward(self, x):
        x = self.mixer_layer(x)
        x = self.ffn_layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, block, depth):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList()
        for _ in range(depth):
            self.encoder.append(block)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        for blk in self.encoder:
            x = blk(x)
        x = self.norm(x)
        return x


class Recommender(nn.Module):
    def __init__(self, n_item, d_model, encoder):
        super(Recommender, self).__init__()
        self.emb_item = nn.Embedding(n_item, d_model, padding_idx=0)
        self.drop = nn.Dropout(0.5)
        self.encoder = encoder
        self.out = nn.Linear(d_model, n_item)

    def forward(self, seq, data_size):
        x = self.drop(self.emb_item(seq))
        encoder_output = self.encoder(x)
        if self.training:
            output = self.out(encoder_output)
        else:
            output = encoder_output[torch.arange(data_size.size(0)), data_size - 1, :].detach()
            output = self.out(output)
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))