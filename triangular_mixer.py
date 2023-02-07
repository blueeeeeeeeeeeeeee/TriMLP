import torch
import torch.nn as nn
from einops import rearrange


def generate_global_mixing(seq_len):
    mask = torch.triu(torch.ones([seq_len, seq_len]))
    matrix = torch.ones([seq_len, seq_len])
    matrix = matrix.masked_fill(mask == 0.0, -1e9)
    kernel = nn.parameter.Parameter(matrix, requires_grad=True)
    return kernel


def generate_local_mixing(seq_len, n_session):
    mask = torch.zeros([seq_len, seq_len])
    for i in range(0, seq_len, seq_len // n_session):
        mask[i:i + seq_len // n_session, i:i + seq_len // n_session] = torch.ones(
            [seq_len // n_session, seq_len // n_session])
    mask = torch.triu(mask)
    matrix = torch.ones([seq_len, seq_len])
    matrix = matrix.masked_fill(mask == 0.0, -1e9)
    kernel = nn.parameter.Parameter(matrix, requires_grad=True)
    return kernel


class TriangularMixer(nn.Module):
    def __init__(self, seq_len, n_session):
        super(TriangularMixer, self).__init__()
        assert seq_len % n_session == 0
        self.act = nn.GELU()
        self.mix_kernel_1 = generate_local_mixing(seq_len, n_session)
        self.mix_kernel_2 = generate_global_mixing(seq_len)

    def forward(self, x):
        x_1 = rearrange(x, 'b n d -> b d n')
        x_1 = self.act(torch.matmul(x_1, self.mix_kernel_1.softmax(dim=-1)))
        x_2 = rearrange(x, 'b n d -> b d n')
        x_2 = self.act(torch.matmul(x_2, self.mix_kernel_2.softmax(dim=-1)))
        x = x_1 + x_2
        x = rearrange(x, 'b d n -> b n d')
        return x