import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.autograd.profiler as profiler
import math
from model.modules.human_graph import CachedGraph

g_dict = {
    'skeleton':CachedGraph.build_human_graph(mode='skeleton'),
    'cayley':CachedGraph.build_human_graph(mode='cayley')
    }

n_additional_node = g_dict['cayley'].num_nodes - g_dict['skeleton'].num_nodes


class SkipableGAT(nn.Module):
    index_of_layer = 0

    def __init__(self, dim:int, drop:float=0.0, use_checkpoint=True, alpha:float=0.1, lamb:float=0.5):
        super().__init__()
        gat_depth:int = 2
        self.use_checkpoint = use_checkpoint
        dr = nn.Dropout(drop * 0.25, True) if drop > 0.001 else nn.Identity()

        self.convs = nn.ModuleList([nn.Sequential(GAT(dim, mode='skeleton', nn.GELU(), dr, nn.LayerNorm(dim))) for _ in range(gat_depth)])
        self.proj_v1 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(gat_depth)])
        self.proj_v2 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(gat_depth)])

        self.alpha = alpha
        self.beta = lamb / (1 + self.__class__.index_of_layer)
        self.__class__.index_of_layer += 1


    def forward(self, x:torch.Tensor):
        if self.training and self.use_checkpoint:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

    
    def _forward_impl(self, x:torch.Tensor):
        # Consider x.shape: [B, T, J, C].
        B, T, J, C = x.shape
        # add virtual nodes.
        x = torch.cat((x, x.new_zeros((B,T,n_additional_node,C))), dim=2)
        x0 = x

        for conv, w1, w2 in zip(self.convs, self.proj_v1, self.proj_v2):
            x = conv(x)
            v1 = (1.0 - self.beta) * x + self.beta * w1(x)
            v2 = (1.0 - self.beta) * x0 + self.beta * w2(x0)
            x = (1.0 - self.alpha) * v1 + self.alpha * v2

        return x[..., :-n_additional_node, :]


class GAT(nn.Module):
    def __init__(self, dim:int, n_heads: int = 4, qk_bias=False, a_scale:int=2, mode:str='skeleton'):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim_h = dim // n_heads
        self.h = n_heads
        self.w_qk = nn.Linear(dim, (2*a_scale)*dim, bias=qkv_bias)
        self.a = nn.Linear(a_scale*self.dim_h, 1, bias=False)
        self.a_scale = a_scale
        self.g = g_dict[mode] 
    
    def forward(self, x:torch.Tensor):
        B, T, J, C = x.shape
        A = self.a_scale*self.dim_h
        # Let start_node[i] = start node of i-th edge.
        start_node, end_node = self.g.edge_index[0], self.g.edge_index[1]
        qk:torch.Tensor = self.w_qk(x)
        qk = qkv.view(B,T,J,self.h,2*A)
        q, k = torch.split(qkv, split_size_or_sections=[A,A], dim=-1) # [B,T,J,H,A]
        z = q[..., start_node,:,:] + k[..., end_node,  :,:] # [B,T,E,H,A]
        z = F.softplus(z)
        z:torch.Tensor = self.a(z).squeeze(-1) # [B,T,E,H]
        z = torch.exp(z - z.amax(dim=2, keepdim=True))
        sigma = x.new_ones(B,T,J,self.h) * 1e-10
        sigma = sigma.index_add(dim=2, index=start_node, source=z) # [B,T,J,H]
        attn = z / sigma[..., start_node, :] # [B,T,E,H]
        attn = attn.transpose(2, 3) # [B,T,H,E]
        dense_attn = x.new_zeros(B,T,self.h,J,J)
        dense_attn[..., start_node, end_node] = attn # [B,T,H,J,J]

        x = x.view(B, T, J, self.h, self.dim_h)
        x = x.transpose(2, 3)
        x = dense_attn @ x
        x = x.transpose(2, 3).reshape(B, T, J, C)
        return x
