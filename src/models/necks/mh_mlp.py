"""
A multi-head MLP neck. Differs from mh_spatial_mlp in that it does not use spatial embeddings directly on the output of the backbone.
"""

import torch
import torch.nn as nn
from einops import rearrange

from .base import BaseNeck
from src.models.layers import init_normalization


class MHMLPNeck(BaseNeck):
    name = 'mh_mlp'

    def __init__(self,
                 in_shape,
                 action_size,
                 num_heads,
                 norm_type,
                 hidden_dims):

        super().__init__(in_shape, action_size)
        c, h, w = self.in_shape
        
        self.base_spatial_embed = nn.Parameter(torch.randn(c * h * w)) # shared across all games
        self.game_offset = nn.Embedding(num_heads, c * h * w) # per game

        self.pool = nn.AvgPool2d((h, w))
        self.norm = init_normalization(channels=c, norm_type=norm_type)

        n_hidden = len(hidden_dims)
        self.mlp = nn.ModuleList()
        for i in range(n_hidden):
            in_dim = c if i == 0 else hidden_dims[i - 1]
            out_dim = hidden_dims[i]
            self.mlp.append(nn.Linear(in_dim, out_dim))
            self.mlp.append(nn.ReLU(inplace=True))


    def forward(self, x, game_id=None):
        """
        Args:
            x (torch.Tensor): (n, t, c, h, w)
            game_id (torch.Tensor): (n, t)
        Returns:
            x (torch.Tensor): (n, t, d)
        """
        n, t, c, h, w = x.shape
        x = rearrange(x, 'n t c h w -> (n t) c h w')

        info = {}

        rep1 = self.pool(x)
        rep1 = rearrange(rep1, 'n c 1 1 -> n c')
        rep1 = rearrange(rep1, '(n t) c -> n t c', n=n, t=t)
        info['rep_candidate_1'] = rep1

        # base: (c * h * w)
        base = rearrange(self.base_spatial_embed, '(c h w) -> 1 c h w', c=c, h=h, w=w)

        if game_id is not None:
            if isinstance(game_id, int):
                game_id = torch.full((n, t), game_id, dtype=torch.long, device=x.device)

            # flatten game_id for embedding lookup
            offset = self.game_offset(game_id)  # (n, t, c*h*w)
            offset = rearrange(offset, 'n t (c h w) -> (n t) c h w', c=c, h=h, w=w)
            spatial_embed = base + offset
        else:
            spatial_embed = base

        x = x * spatial_embed

        x = self.pool(x)
        x = rearrange(x, 'n c 1 1 -> n c')
        x = self.norm(x)  # type: ignore

        rep2 = rearrange(x, '(n t) c -> n t c', n=n, t=t)
        info['rep_candidate_2'] = rep2

        for layer in self.mlp:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                rep = rearrange(x, '(n t) d -> n t d', n=n, t=t)
                info_key = f'rep_candidate_{len(info) + 1}'
                info[info_key] = rep

        x = rearrange(x, '(n t) d -> n t d', n=n, t=t)

        return x, info
