import torch
from torch import nn
from torch.fft import fft2, ifft2

from nn_blocks import LayerNorm, ResnetBlock

class Conditioning(nn.Module):
    def __init__(self, fmap_size, dim):
        super().__init__()
        self.ff_parser_attn_map = nn.Parameter(torch.ones(dim, fmap_size, fmap_size))

        self.norm_input = LayerNorm(dim, bias = True)
        self.norm_condition = LayerNorm(dim, bias = True)

        self.block = ResnetBlock(dim, dim)

    def forward(self, x, c):

        # ff-parser in the paper, for modulating out the high frequencies

        dtype = x.dtype
        x = fft2(x)
        x = x * self.ff_parser_attn_map
        x = ifft2(x).real
        x = x.type(dtype)

        # eq 3 in paper

        normed_x = self.norm_input(x)
        normed_c = self.norm_condition(c)
        c = (normed_x * normed_c) * c

        # add an extra block to allow for more integration of information
        # there is a downsample right after the Condition block (but maybe theres a better place to condition than right before the downsample)

        return self.block(c)
