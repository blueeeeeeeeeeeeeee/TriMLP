import torch
import torch.nn as nn
from einops import rearrange


# Square Encoder--------------------------------------------------------------------------------------------------------
class Square(nn.Module):
    def __init__(self, seq_len):
        super(Square, self).__init__()
        self.act = nn.GELU()
        self.mix_kernel = nn.Linear(seq_len, seq_len, bias=False)

    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.act(self.mix_kernel(x))
        x = rearrange(x, 'b d n -> b n d')
        return x
# ----------------------------------------------------------------------------------------------------------------------


# Eye Encoder ----------------------------------------------------------------------------------------------------------
class Eye(nn.Module):
    def __init__(self, seq_len):
        super(Eye, self).__init__()
        self.act = nn.GELU()
        mask = torch.eye(seq_len)
        matrix = torch.eye(seq_len)
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel = nn.parameter.Parameter(matrix, requires_grad=True)

    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.act(torch.matmul(x, self.mix_kernel.softmax(dim=-1)))
        x = rearrange(x, 'b d n -> b n d')
        return x
# ----------------------------------------------------------------------------------------------------------------------


# Tri Encoder-----------------------------------------------------------------------------------------------------------
class Tri(nn.Module):
    def __init__(self, seq_len):
        super(Tri, self).__init__()
        self.act = nn.GELU()
        mask = torch.triu(torch.ones([seq_len, seq_len]))
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel = nn.parameter.Parameter(matrix, requires_grad=True)

    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.act(torch.matmul(x, self.mix_kernel.softmax(dim=-1)))
        x = rearrange(x, 'b d n -> b n d')
        return x
# ----------------------------------------------------------------------------------------------------------------------


# Multi-Head Tri Encoder------------------------------------------------------------------------------------------------
class MHTri(nn.Module):
    def __init__(self, seq_len, d_model, n_head):
        super(MHTri, self).__init__()
        assert d_model % n_head == 0
        self.act = nn.GELU()
        self.d_k = d_model // n_head
        self.n_h = n_head
        mask = torch.triu(torch.ones([seq_len, seq_len]))
        mask = mask.unsqueeze(0)
        matrix = torch.ones([n_head, seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel = nn.parameter.Parameter(matrix, requires_grad=True)
        self.map = nn.Linear(d_model, d_model)
        self.merge = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.map(x)
        x = rearrange(x, 'b n (h d_k) -> b h d_k n', d_k=self.d_k)
        x = self.act(torch.matmul(x, self.mix_kernel.softmax(dim=-1)))
        x = rearrange(x, 'b h d_k n -> b n (h d_k)', d_k=self.d_k)
        x = self.merge(x)
        return x
# ----------------------------------------------------------------------------------------------------------------------


# Multi-Segment Tri Encoder---------------------------------------------------------------------------------------------
class MSTri(nn.Module):
    def __init__(self, seq_len, n_segment):
        super(MSTri, self).__init__()
        assert seq_len % n_segment == 0
        self.act = nn.GELU()
        mask = torch.zeros([seq_len, seq_len])
        for i in range(0, seq_len, seq_len//n_segment):
            mask[i:i+seq_len//n_segment, i:i+seq_len//n_segment] = torch.ones([seq_len//n_segment, seq_len//n_segment])
        mask = torch.triu(mask)
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel = nn.parameter.Parameter(matrix, requires_grad=True)

    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.act(torch.matmul(x, self.mix_kernel.softmax(dim=-1)))
        x = rearrange(x, 'b d n -> b n d')
        return x
# -----------------------------------------------------------------------------------------------------------------------


# Multi-Segment Tri Encoder (OverLap)-----------------------------------------------------------------------------------
class MSTriOL(nn.Module):
    def __init__(self, seq_len, n_segment):
        super(MSTriOL, self).__init__()
        assert seq_len % n_segment == 0
        self.act = nn.GELU()
        mask = torch.ones([seq_len, seq_len])
        lower_mask = torch.tril(mask, diagonal=-1)
        upper_mask = torch.triu(mask, diagonal=seq_len//n_segment)
        mask = lower_mask + upper_mask
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 1.0, -1e9)
        self.mix_kernel = nn.parameter.Parameter(matrix, requires_grad=True)

    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.act(torch.matmul(x, self.mix_kernel.softmax(dim=-1)))
        x = rearrange(x, 'b d n -> b n d')
        return x
# -----------------------------------------------------------------------------------------------------------------------


# MSTri + Tri-----------------------------------------------------------------------------------------------------------
class Plus(nn.Module):
    def __init__(self, seq_len, n_segment):
        super(Plus, self).__init__()
        assert seq_len % n_segment == 0
        self.act = nn.GELU()
        mask = torch.zeros([seq_len, seq_len])
        for i in range(0, seq_len, seq_len // n_segment):
            mask[i:i + seq_len // n_segment, i:i + seq_len // n_segment] = torch.ones(
                [seq_len // n_segment, seq_len // n_segment])
        mask = torch.triu(mask)
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel_1 = nn.parameter.Parameter(matrix, requires_grad=True)

        mask = torch.triu(torch.ones([seq_len, seq_len]))
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel_2 = nn.parameter.Parameter(matrix, requires_grad=True)

    def forward(self, x):
        x_1 = rearrange(x, 'b n d -> b d n')
        x_1 = self.act(torch.matmul(x_1, self.mix_kernel_1.softmax(dim=-1)))
        x_2 = rearrange(x, 'b n d -> b d n')
        x_2 = self.act(torch.matmul(x_2, self.mix_kernel_2.softmax(dim=-1)))
        x = x_1 + x_2
        x = rearrange(x, 'b d n -> b n d')
        return x
# -----------------------------------------------------------------------------------------------------------------------


# MSTri || Tri---------------------------------------------------------------------------------------------------------
class Merge(nn.Module):
    def __init__(self, seq_len, n_segment, d_model):
        super(Merge, self).__init__()
        assert seq_len % n_segment == 0
        self.act = nn.GELU()
        mask = torch.zeros([seq_len, seq_len])
        for i in range(0, seq_len, seq_len // n_segment):
            mask[i:i + seq_len // n_segment, i:i + seq_len // n_segment] = torch.ones(
                [seq_len // n_segment, seq_len // n_segment])
        mask = torch.triu(mask)
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel_1 = nn.parameter.Parameter(matrix, requires_grad=True)

        mask = torch.triu(torch.ones([seq_len, seq_len]))
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel_2 = nn.parameter.Parameter(matrix, requires_grad=True)

        self.merge = nn.Linear(2*d_model, d_model)

    def forward(self, x):
        x_1 = rearrange(x, 'b n d -> b d n')
        x_1 = self.act(torch.matmul(x_1, self.mix_kernel_1.softmax(dim=-1)))
        x_2 = rearrange(x, 'b n d -> b d n')
        x_2 = self.act(torch.matmul(x_2, self.mix_kernel_2.softmax(dim=-1)))
        x = torch.cat([x_1, x_2], dim=1)
        x = rearrange(x, 'b d n -> b n d')
        x = self.merge(x)
        return x
# -----------------------------------------------------------------------------------------------------------------------


# MSTri -> Tri----------------------------------------------------------------------------------------------------------
class Serial(nn.Module):
    def __init__(self, seq_len, n_segment):
        super(Serial, self).__init__()
        assert seq_len % n_segment == 0
        self.act = nn.GELU()
        mask = torch.zeros([seq_len, seq_len])
        for i in range(0, seq_len, seq_len // n_segment):
            mask[i:i + seq_len // n_segment, i:i + seq_len // n_segment] = torch.ones(
                [seq_len // n_segment, seq_len // n_segment])
        mask = torch.triu(mask)
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel_1 = nn.parameter.Parameter(matrix, requires_grad=True)

        mask = torch.triu(torch.ones([seq_len, seq_len]))
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel_2 = nn.parameter.Parameter(matrix, requires_grad=True)

    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.act(torch.matmul(x, self.mix_kernel_1.softmax(dim=-1)))
        x = self.act(torch.matmul(x, self.mix_kernel_2.softmax(dim=-1)))
        x = rearrange(x, 'b d n -> b n d')
        return x
# -----------------------------------------------------------------------------------------------------------------------


# Tri -> MSTri----------------------------------------------------------------------------------------------------------
class SerialR(nn.Module):
    def __init__(self, seq_len, n_segment):
        super(SerialR, self).__init__()
        assert seq_len % n_segment == 0
        self.act = nn.GELU()
        mask = torch.zeros([seq_len, seq_len])
        for i in range(0, seq_len, seq_len // n_segment):
            mask[i:i + seq_len // n_segment, i:i + seq_len // n_segment] = torch.ones(
                [seq_len // n_segment, seq_len // n_segment])
        mask = torch.triu(mask)
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel_1 = nn.parameter.Parameter(matrix, requires_grad=True)

        mask = torch.triu(torch.ones([seq_len, seq_len]))
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel_2 = nn.parameter.Parameter(matrix, requires_grad=True)

    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.act(torch.matmul(x, self.mix_kernel_2.softmax(dim=-1)))
        x = self.act(torch.matmul(x, self.mix_kernel_1.softmax(dim=-1)))
        x = rearrange(x, 'b d n -> b n d')
        return x
# -----------------------------------------------------------------------------------------------------------------------


# MSTri + Tri OverLap---------------------------------------------------------------------------------------------------
class PlusOL(nn.Module):
    def __init__(self, seq_len, n_segment):
        super(PlusOL, self).__init__()
        assert seq_len % n_segment == 0
        self.act = nn.GELU()
        mask = torch.ones([seq_len, seq_len])
        lower_mask = torch.tril(mask, diagonal=-1)
        upper_mask = torch.triu(mask, diagonal=seq_len // n_segment)
        mask = lower_mask + upper_mask
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 1.0, -1e9)
        self.mix_kernel_1 = nn.parameter.Parameter(matrix, requires_grad=True)

        mask = torch.triu(torch.ones([seq_len, seq_len]))
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel_2 = nn.parameter.Parameter(matrix, requires_grad=True)

    def forward(self, x):
        x_1 = rearrange(x, 'b n d -> b d n')
        x_1 = self.act(torch.matmul(x_1, self.mix_kernel_1.softmax(dim=-1)))
        x_2 = rearrange(x, 'b n d -> b d n')
        x_2 = self.act(torch.matmul(x_2, self.mix_kernel_2.softmax(dim=-1)))
        x = x_1 + x_2
        x = rearrange(x, 'b d n -> b n d')
        return x
# -----------------------------------------------------------------------------------------------------------------------


# MSTri || Tri OverLap--------------------------------------------------------------------------------------------------
class MergeOL(nn.Module):
    def __init__(self, seq_len, n_segment, d_model):
        super(MergeOL, self).__init__()
        assert seq_len % n_segment == 0
        self.act = nn.GELU()
        mask = torch.ones([seq_len, seq_len])
        lower_mask = torch.tril(mask, diagonal=-1)
        upper_mask = torch.triu(mask, diagonal=seq_len // n_segment)
        mask = lower_mask + upper_mask
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 1.0, -1e9)
        self.mix_kernel_1 = nn.parameter.Parameter(matrix, requires_grad=True)

        mask = torch.triu(torch.ones([seq_len, seq_len]))
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel_2 = nn.parameter.Parameter(matrix, requires_grad=True)

        self.merge = nn.Linear(2*d_model, d_model)

    def forward(self, x):
        x_1 = rearrange(x, 'b n d -> b d n')
        x_1 = self.act(torch.matmul(x_1, self.mix_kernel_1.softmax(dim=-1)))
        x_2 = rearrange(x, 'b n d -> b d n')
        x_2 = self.act(torch.matmul(x_2, self.mix_kernel_2.softmax(dim=-1)))
        x = torch.cat([x_1, x_2], dim=1)
        x = rearrange(x, 'b d n -> b n d')
        x = self.merge(x)
        return x
# -----------------------------------------------------------------------------------------------------------------------


# MSTri -> Tri OverLap--------------------------------------------------------------------------------------------------
class SerialOL(nn.Module):
    def __init__(self, seq_len, n_segment):
        super(SerialOL, self).__init__()
        assert seq_len % n_segment == 0
        self.act = nn.GELU()
        mask = torch.ones([seq_len, seq_len])
        lower_mask = torch.tril(mask, diagonal=-1)
        upper_mask = torch.triu(mask, diagonal=seq_len // n_segment)
        mask = lower_mask + upper_mask
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 1.0, -1e9)
        self.mix_kernel_1 = nn.parameter.Parameter(matrix, requires_grad=True)

        mask = torch.triu(torch.ones([seq_len, seq_len]))
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel_2 = nn.parameter.Parameter(matrix, requires_grad=True)

    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.act(torch.matmul(x, self.mix_kernel_1.softmax(dim=-1)))
        x = self.act(torch.matmul(x, self.mix_kernel_2.softmax(dim=-1)))
        x = rearrange(x, 'b d n -> b n d')
        return x
# -----------------------------------------------------------------------------------------------------------------------


# Tri -> MSTri OverLap--------------------------------------------------------------------------------------------------
class SerialROL(nn.Module):
    def __init__(self, seq_len, n_segment):
        super(SerialROL, self).__init__()
        assert seq_len % n_segment == 0
        self.act = nn.GELU()
        mask = torch.ones([seq_len, seq_len])
        lower_mask = torch.tril(mask, diagonal=-1)
        upper_mask = torch.triu(mask, diagonal=seq_len // n_segment)
        mask = lower_mask + upper_mask
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 1.0, -1e9)
        self.mix_kernel_1 = nn.parameter.Parameter(matrix, requires_grad=True)

        mask = torch.triu(torch.ones([seq_len, seq_len]))
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.mix_kernel_2 = nn.parameter.Parameter(matrix, requires_grad=True)

    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.act(torch.matmul(x, self.mix_kernel_2.softmax(dim=-1)))
        x = self.act(torch.matmul(x, self.mix_kernel_1.softmax(dim=-1)))
        x = rearrange(x, 'b d n -> b n d')
        return x
# -----------------------------------------------------------------------------------------------------------------------
