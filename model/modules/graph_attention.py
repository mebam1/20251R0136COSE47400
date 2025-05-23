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
    def __init__(self, dim:int, drop:float=0.0, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        dr = nn.Dropout(drop * 0.25, True) if drop > 0.001 else nn.Identity()
        conv1 = nn.Sequential(GAT(dim, mode='skeleton'), dr, nn.LayerNorm(dim))
        conv2 = nn.Sequential(GAT(dim, mode='cayley'), dr, nn.LayerNorm(dim))
        conv3 = nn.Sequential(GAT(dim, mode='skeleton'), dr, nn.LayerNorm(dim))
        conv4 = nn.Sequential(GAT(dim, mode='cayley'), dr, nn.LayerNorm(dim))
        self.convs = nn.ModuleList([conv1, conv2, conv3, conv4])
        self.proj = nn.Sequential(nn.Linear(dim*(1 + len(self.convs)), dim), nn.LayerNorm(dim))

    def forward(self, x:torch.Tensor):
        if self.training and self.use_checkpoint:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

    
    def _forward_impl(self, x:torch.Tensor):
        # Consider x.shape: [B, T, J, C].
        B, T, J, C = x.shape
        outputs = [x]
        x = torch.cat((x, x.new_zeros((B,T,n_additional_node,C))), dim=2)

        for conv in self.convs:
            x = conv(x)
            outputs.append(x[..., :-n_additional_node,:])
        x = torch.cat(outputs, dim=-1)
        x = self.proj(x)
        return x
    

class Alpha(nn.Module):
    def __init__(self, dim:int, p_x:float=0.2):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(2*dim, dim), nn.LayerNorm(dim), nn.Softplus(), nn.Linear(dim, 2), nn.Softmax(dim=-1))
        z = math.log((1.0 - p_x) / p_x) * 0.5
        with torch.no_grad():
            self.out[3].bias.copy_(torch.tensor([-z, z]))


    def forward(self, x:torch.Tensor, y:torch.Tensor):
        return self.out(torch.cat((x, y), dim=-1))


class GAT(nn.Module):
    def __init__(self, dim:int, n_heads: int = 8, qkv_bias=False, a_scale:int=2, mode:str='skeleton', beta:float=0.8):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        assert 0.0 <= beta and beta <= 1.0, "beta must be on [0.0, 1.0]"
        self.dim_h = dim // n_heads
        self.h = n_heads
        self.w_qkv = nn.Linear(dim, (3*a_scale)*dim, bias=qkv_bias)
        self.a = nn.Linear(a_scale*self.dim_h, 1, bias=False)
        self.a_scale = a_scale
        self.g = g_dict[mode] 
        self.beta = beta
    
    def forward(self, x:torch.Tensor):
        B, T, J, C = x.shape
        A = self.a_scale*self.dim_h
        # Let start_node[i] = start node of i-th edge.
        start_node, end_node = self.g.edge_index[0], self.g.edge_index[1]
        qkv:torch.Tensor = self.w_qkv(x)
        qkv = qkv.view(B,T,J,self.h,3*A)
        q, k, v = torch.split(qkv, split_size_or_sections=[A,A,A], dim=-1) # [B,T,J,H,A]
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
        v = v.transpose(2, 3) # [B,T,H,J,A]
        v1, v2 = torch.split(v, split_size_or_sections=[self.dim_h, self.dim_h], dim=-1) # [B,T,H,J,dH]
        v = (1.0 - self.beta) * v1 + self.beta * dense_attn @ v2 # [B,T,H,J,dH]
        v = v.transpose(2, 3).reshape(B,T,J,C)
        return x + v



#==================#
#  UNIT TEST CODE  #
#==================#

def show_mem(tag=""):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserv = torch.cuda.memory_reserved() / 1024**2
    print(f"[{tag}] allocated = {alloc:.1f} MB | reserved = {reserv:.1f} MB")
    print(tag)

def profile_gat():
    B, T, J, C = 8, 81, 17, 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GAT(C,C,4)
    model.to(device)
    # Test forward pass
    x = torch.randn(B, T, J, C, device=device)
    sort_by = 'cuda_memory_usage'
    with profiler.profile(with_stack=True, 
                          profile_memory=True, 
                          use_device='cuda',
                          experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        x = model(x, j_graph)
    torch.cuda.synchronize()
    print(prof.key_averages(group_by_stack_n=8).table(sort_by=sort_by, row_limit=10))

def profile_skip_gat():
    B, T, J, C = 8, 81, 17, 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SkipableGAT(C, 4, drop=0.4)
    model.to(device)
    # Test forward pass
    x = torch.randn(B, T, J, C, device=device)
    sort_by = 'cuda_memory_usage'
    with profiler.profile(with_stack=True, 
                          profile_memory=True, 
                          use_device='cuda',
                          experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        x = model(x, j_graph)
    torch.cuda.synchronize()
    print(prof.key_averages(group_by_stack_n=8).table(sort_by=sort_by, row_limit=10))
    

def test_many_modules():
    B, T, J, C = 8, 81, 17, 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = nn.ModuleList([SkipableGAT(dim=C, drop=0.1, bottle_neck=2) for _ in range(16)])
    model.to(device)

    x = torch.randn(B, T, J, C, device=device)
    show_mem('---before---')
    for m in model:
        x = m(x)
    show_mem('---after---')

if __name__ == '__main__':
    print(j_graph.edge_index.shape)
    x  =j_graph.edge_index
    for i in range(x.shape[1]):
        print(x[0, i].item(), ' ' ,x[1, i].item())
    
