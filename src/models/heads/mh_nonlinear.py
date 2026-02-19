
from .base import BaseHead
import torch
import torch.nn as nn
import math
from einops import rearrange


class MHNonLinearHead(BaseHead):
    name = 'mh_nonlinear'
    def __init__(self,
                in_shape,
                hidden_sizes,
                action_size,
                num_heads,
                activation='relu'):
        
        super().__init__(in_shape, action_size)
        self.num_heads = num_heads
        self.in_dim = in_shape if isinstance(in_shape, int) else in_shape[0]
        self.hidden_sizes = list(hidden_sizes)

        self.activation = nn.ReLU() if activation == 'relu' else None
        if self.activation is None:
            raise ValueError("Missing activation function for MHNonLinearHead")

        # print(f"Hidden sizes for MHNonLinearHead: {self.hidden_sizes}")

        layer_dims = [self.in_dim] + self.hidden_sizes + [action_size]

        self.layer_weights = nn.ModuleList()
        self.layer_biases = nn.ModuleList()

        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i+1]
            self.layer_weights.append(nn.Embedding(num_heads, in_dim * out_dim))
            self.layer_biases.append(nn.Embedding(num_heads, out_dim))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for i, (weight_emb, bias_emb) in enumerate(zip(self.layer_weights, self.layer_biases)):
            if i == 0:
                fan_in = self.in_dim
            else:
                fan_in = self.hidden_sizes[i - 1]
            
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(weight_emb.weight, -bound, bound)
            nn.init.uniform_(bias_emb.weight, -bound, bound)
        
    def forward(self, x, game_id=None):
        """
        [params] x (torch.Tensor: (n, t, d))
        [params] game_id (torch.Tensor: (n, t)) game_id of each head to utilize
        [returns] x (torch.Tensor: (n, t, a))
        """
        n, t, d = x.shape
        if game_id is None:
            game_id = torch.zeros((n, t), device=x.device).long()
        
        # Flatten batch and time dimensions
        x = rearrange(x, 'n t d -> (n t) d')
        game_id_flat = rearrange(game_id, 'n t -> (n t)')
        
        # Pass through each layer
        layer_dims = [self.in_dim] + self.hidden_sizes + [self.action_size]
        for i, (weight_emb, bias_emb) in enumerate(zip(self.layer_weights, self.layer_biases)):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            
            # Get weights and biases for this layer
            weights = weight_emb(game_id_flat)  # (n*t, in_dim * out_dim)
            biases = bias_emb(game_id_flat)      # (n*t, out_dim)
            
            # Reshape weights for batch matrix multiplication
            weights = rearrange(weights, 'b (i o) -> b i o', i=in_dim, o=out_dim)
            
            # Apply linear transformation
            x = torch.bmm(x.unsqueeze(1), weights).squeeze(1) + biases
            
            # Apply activation (except for last layer)
            if i < len(self.layer_weights) - 1:
                x = self.activation(x)
        
        # Reshape back to (n, t, a)
        x = rearrange(x, '(n t) a -> n t a', t=t)
        
        info = {}
        return x, info
