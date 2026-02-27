""" A multi-head nonlinear distributional head. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

from .base import BaseHead

class MHNonLinearDistributionalHead(BaseHead):
    name = "mh_nonlinear_distributional"
    def __init__(self, in_shape, hidden_sizes, action_size, num_heads, num_atoms, activation='relu', **kwargs):
        super().__init__(in_shape, action_size)
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        self.in_dim = in_shape if isinstance(in_shape, int) else in_shape[0]
        self.hidden_sizes = list(hidden_sizes)

        # Choose activation
        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}
        self.activation = activations.get(activation)
        if self.activation is None:
            raise ValueError(f"Unsupported activation: {activation}")

        # Hidden layers (per-head embeddings)
        layer_dims = [self.in_dim] + self.hidden_sizes
        self.hidden_weights = nn.ModuleList()
        self.hidden_biases = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.hidden_weights.append(nn.Embedding(num_heads, layer_dims[i] * layer_dims[i + 1]))
            self.hidden_biases.append(nn.Embedding(num_heads, layer_dims[i + 1]))

        # Final layer (action x num_atoms)
        last_hidden_dim = layer_dims[-1] if self.hidden_sizes else self.in_dim
        self.output_weights = nn.Embedding(num_heads, last_hidden_dim * self.action_size * num_atoms)
        self.output_biases = nn.Embedding(num_heads, self.action_size * num_atoms)

        self.reset_parameters()

    def reset_parameters(self):
        # Hidden layers
        layer_dims = [self.in_dim] + self.hidden_sizes
        for i, (w_emb, b_emb) in enumerate(zip(self.hidden_weights, self.hidden_biases)):
            fan_in = layer_dims[i]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(w_emb.weight, -bound, bound)
            nn.init.uniform_(b_emb.weight, -bound, bound)

        # Output layer
        fan_in = layer_dims[-1] if self.hidden_sizes else self.in_dim
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.output_weights.weight, -bound, bound)
        nn.init.uniform_(self.output_biases.weight, -bound, bound)

    def forward(self, x, idx=None):
        """
        :param x (torch.Tensor): (n, t, d)
        :param idx (torch.Tensor): (n, t) head index
        :returns: x -> (n, t, action_size, num_atoms)
        """
        n, t, _ = x.shape
        a, n_a = self.action_size, self.num_atoms
        if idx is None:
            idx = torch.zeros((n, t), device=x.device).long()

        x = rearrange(x, 'n t d -> (n t) d')
        idx_flat = rearrange(idx, 'n t -> (n t)')

        # Pass through hidden layers
        layer_dims = [self.in_dim] + self.hidden_sizes
        for i, (w_emb, b_emb) in enumerate(zip(self.hidden_weights, self.hidden_biases)):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            w = rearrange(w_emb(idx_flat), 'b (i o) -> b i o', i=in_dim, o=out_dim)
            b = b_emb(idx_flat)
            x = torch.bmm(x.unsqueeze(1), w).squeeze(1) + b
            x = self.activation(x) # type: ignore

        # Final linear layer -> action x num_atoms
        b = idx_flat.shape[0]
        w_out = self.output_weights(idx_flat).view(b, x.shape[1], a * n_a)

        b_out = self.output_biases(idx_flat)
        x = torch.bmm(x.unsqueeze(1), w_out).squeeze(1) + b_out

        # Reshape to (n, t, action_size, num_atoms) and apply softmax over atoms
        nt = x.shape[0]
        x = x.view(nt // t, t, a, n_a)
        log_x = F.log_softmax(x, dim=-1)
        return torch.exp(log_x), {'log': log_x}
