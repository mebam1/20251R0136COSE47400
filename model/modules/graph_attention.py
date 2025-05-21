import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.autograd.profiler as profiler
import math


class CachedGraph():
    @torch.no_grad()
    def __init__(self, edge_index:torch.Tensor, num_nodes:int):

        one_hop_adj = CachedGraph.get_adj(num_nodes, edge_index)
        two_hop_adj = one_hop_adj @ one_hop_adj
        adj = one_hop_adj + two_hop_adj
        adj.fill_diagonal_(0)
        self.edge_index = self.get_edge_index(adj)

        g1, g2 = num_nodes, num_nodes + 1
        all_base_nodes = torch.arange(17, device=edge_index.device)
        edges_global1 = torch.stack([all_base_nodes, torch.full_like(all_base_nodes, g1)], dim=0)
        edges_global1_rev = torch.stack([torch.full_like(all_base_nodes, g1), all_base_nodes], dim=0)
        edges_global2 = torch.stack([all_base_nodes, torch.full_like(all_base_nodes, g2)], dim=0)
        edges_global2_rev = torch.stack([torch.full_like(all_base_nodes, g2), all_base_nodes], dim=0)
        self.edge_index = torch.cat([
            self.edge_index,
            edges_global1, edges_global1_rev,
            edges_global2, edges_global2_rev
        ], dim=1)

        self.num_nodes = num_nodes + 2


    @staticmethod
    @torch.no_grad()
    def get_adj(num_nodes:int, edge_index:torch.Tensor) -> torch.Tensor:
        adj = edge_index.new_zeros((num_nodes, num_nodes), dtype=torch.float32)
        src, dst = edge_index[0], edge_index[1]
        adj[src, dst] = 1.0
        return adj
    
    @staticmethod
    @torch.no_grad()
    def get_edge_index(adj:torch.Tensor) -> torch.Tensor:
        src, dst = adj.nonzero(as_tuple=True)
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index


    @staticmethod
    @torch.no_grad()
    def build_human_graph():

        device ='cuda' if torch.cuda.is_available() else 'cpu'
        x = torch.tensor([
            [
                0,0,0,1,1,2,2,3,4,4,5,5,6,7,7,8,8,8,8,9,9,10,11,11,12,12,13,14,14,15,15,16
            ],
            [
                1,4,7,0,2,1,3,2,0,5,4,6,5,0,8,7,9,11,14,8,10,9,8,12,11,13,12,8,15,14,16,15
            ]
            ], device=device, dtype=torch.long)
        
        return CachedGraph(x, 17)
    

j_graph=CachedGraph.build_human_graph()

class SkipableGAT(nn.Module):
    def __init__(self, dim:int, bottle_neck:int=2, drop:float=0.0, use_checkpoint=True):
        super().__init__()
        assert dim % bottle_neck == 0, f"(dim={dim}) must be divisible by (bottleneck={bottle_neck})."
        gat_dim = dim // bottle_neck
        self.use_checkpoint = use_checkpoint
        conv1 = nn.Sequential(nn.Linear(dim, gat_dim), GAT(gat_dim), nn.Linear(gat_dim, dim))
        norm = nn.LayerNorm(dim)
        conv2 = nn.Sequential(nn.Linear(dim, gat_dim), GAT(gat_dim), nn.Linear(gat_dim, dim))

        self.gat = nn.Sequential(
            conv1,
            nn.Dropout(drop * 0.5, True) if drop < 0.001 else nn.Identity(),
            norm,
            conv2,
            nn.Dropout(drop * 0.5, True) if drop < 0.001 else nn.Identity()
        )

    def forward(self, x:torch.Tensor):
        if self.training and self.use_checkpoint:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

    
    def _forward_impl(self, x:torch.Tensor):
        # Consider x.shape: [B, T, J, C].
        x_mean_global = x.mean(dim=2, keepdim=True) # [B, T, 1, C]
        x_zero_global = torch.zeros_like(x_mean_global)
        x = torch.cat([x, x_zero_global, x_mean_global], dim=2) # [B, T, J+2, C], append global feature.
        x = self.gat(x)
        return x[..., :-2, :]
    

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
    def __init__(self, dim:int, n_heads: int = 8, qkv_bias=False, a_scale:int=2):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim_h = dim // n_heads
        self.h = n_heads
        self.w_qkv = nn.Linear(dim, (3*a_scale)*dim, bias=qkv_bias)
        self.a = nn.Linear(a_scale*self.dim_h, 1, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.a_scale = a_scale
        self.g = j_graph
    
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
        v = v1 + dense_attn @ v2 # [B,T,H,J,dH]
        v = v.transpose(2, 3).reshape(B,T,J,C)
        return v



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
    
