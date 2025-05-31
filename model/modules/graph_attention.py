import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as linalg
from torch.utils.checkpoint import checkpoint
from model.modules.human_graph import CachedGraph
import math

g_dict = {
    'skeleton':CachedGraph.build_human_graph(mode='skeleton'),
    'cayley':CachedGraph.build_human_graph(mode='cayley')
    }

n_additional_node = g_dict['cayley'].num_nodes - g_dict['skeleton'].num_nodes


class SkipableGAT(nn.Module):
    index_of_layer = 0

    def __init__(self, dim:int, drop:float=0.0, use_checkpoint=True, alpha:float=0.1, lamb:float=0.5):
        super().__init__()
        gat_depth:int = 1
        self.use_checkpoint = use_checkpoint
        dr = nn.Dropout(drop * 0.25, True) if drop > 0.001 else nn.Identity()

        self.convs = nn.ModuleList([GAT(dim, mode='skeleton') for _ in range(gat_depth)])
        self.proj_v1 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(gat_depth)])
        self.proj_v2 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(gat_depth)])

        self.alpha = alpha
        self.beta = lamb / (1 + self.__class__.index_of_layer)
        self.__class__.index_of_layer += 1

        for p, q in zip(self.proj_v1, self.proj_v2):
            nn.init.xavier_normal_(p.weight)
            nn.init.xavier_normal_(q.weight)


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
            x = F.gelu(x)

        return x[..., :-n_additional_node, :]


class GAT(nn.Module):
    def __init__(self, dim:int, n_heads: int = 8, qk_bias=False, a_scale:int=2, mode:str='skeleton'):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim_h = dim // n_heads
        self.h = n_heads
        self.w_qk = nn.Linear(dim, (2*a_scale)*dim, bias=qk_bias)
        self.a = nn.Parameter(torch.empty(self.h, self.dim_h * a_scale), requires_grad=True)
        self.a_scale = a_scale
        self.g = g_dict[mode] 
        self.init_params()

    def init_params(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        nn.init.xavier_normal_(self.a, gain=gain)
    
    def forward(self, x:torch.Tensor):
        B, T, J, C = x.shape
        A = self.a_scale*self.dim_h
        # Let start_node[i] = start node of i-th edge.
        start_node, end_node = self.g.edge_index[0], self.g.edge_index[1]
        qk:torch.Tensor = self.w_qk(x)
        qk = qk.view(B,T,J,self.h,2*A)
        q, k = torch.split(qk, split_size_or_sections=[A,A], dim=-1) # [B,T,J,H,A]
        z = q[..., start_node,:,:] + k[..., end_node,  :,:] # [B,T,E,H,A]
        z = F.leaky_relu_(z, negative_slope=0.2)
        z = torch.einsum('bteha,ha->bteh', z, self.a)
        z = torch.exp(z - z.amax(dim=2, keepdim=True))
        sigma = x.new_zeros(B,T,J,self.h)
        sigma = sigma.index_add(dim=2, index=end_node, source=z) # [B,T,J,H]
        attn = z / (sigma[..., end_node, :] + 1e-9) # [B,T,E,H]
        attn = attn.transpose(2, 3) # [B,T,H,E]
        dense_attn = x.new_zeros(B,T,self.h,J,J)
        dense_attn[..., end_node, start_node] = attn # [B,T,H,J,J]

        x = x.view(B, T, J, self.h, self.dim_h)
        x = x.transpose(2, 3)
        x = dense_attn @ x
        x = x.transpose(2, 3).reshape(B, T, J, C)
        return x

def _test():
    dim=128
    dev = 'cuda'
    gat = SkipableGAT(dim, use_checkpoint=False)
    gat.to(dev)
    x = torch.zeros(size=(1, 1, 17, 128), device=dev)

    for i in range(17):
        x[0, 0, i, i] = 1.0

    y = gat(x)
    #print(y.shape)
    a = y[0, 0, 0, :]
    b = y[0, 0, 6, :]
    print(a)
    print(b)

    print('---')
    print((torch.sum(a * b, dim=-1) / linalg.norm(a) / linalg.norm(b)).item())



if __name__ == '__main__':
    _test()