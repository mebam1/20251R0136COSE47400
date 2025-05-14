import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.autograd.profiler as profiler


class CachedGraph():
    @torch.no_grad()
    def __init__(self, edge_index:torch.Tensor, num_nodes:int):
        self.num_nodes = num_nodes
        self.edge_index = edge_index

    @staticmethod
    @torch.no_grad()
    def build_human_graph(joint_or_bone:str):
        if joint_or_bone not in ['joint', 'bone']:
            raise ValueError("joint_or_bone must be 'joint' or 'bone'")
        device ='cuda' if torch.cuda.is_available() else 'cpu'

        if joint_or_bone == 'joint':
            # H36M Joint structure
            x = torch.tensor([
                [
                    0,0,0,1,1,2,2,3,4,4,5,5,6,7,7,8,8,8,8,9,9,10,11,11,12,12,13,14,14,15,15,16
                ],
                [
                    1,4,7,0,2,1,3,2,0,5,4,6,5,0,8,7,9,11,14,8,10,9,8,12,11,13,12,8,15,14,16,15
                ]
                ], device=device, dtype=torch.long)
            
            #x = x[:, (x[0] < x[1])]
            
            x = torch.cat([x, torch.arange(0, 17, step=1,device=device, dtype=torch.int64).expand(2, -1)], dim=1)
            return CachedGraph(x, 17)
        else:
            # Human Bone structure (DSTFormer)
            x = torch.tensor([
                [
                    0,0,0,1,1,2,3,3,3,4,4,5,6,6,6,7,7,7,7,8,8,8,8,9,10,10,10,10,11,11,12,13,13,13,13,14,14,15
                ],
                [
                    1,3,6,0,2,1,0,4,6,3,5,4,0,3,7,6,8,10,13,7,9,10,13,8,7,8,11,13,10,12,11,7,8,10,14,13,15,14
                ]
                ], device=device, dtype=torch.long)
            #x = x[:, (x[0] < x[1])]
            x = torch.cat([x, torch.arange(0, 16, step=1,device=device, dtype=torch.int64).expand(2, -1)], dim=1)
            return CachedGraph(x, 16)

j_graph=CachedGraph.build_human_graph('joint')
b_graph=CachedGraph.build_human_graph('bone')

class SkipableGAT(nn.Module):
    def __init__(self, dim:int, bottle_neck:int=2, drop:float=0.0, use_checkpoint=True):
        super().__init__()
        assert dim % bottle_neck == 0, f"(dim={dim}) must be divisible by (bottleneck={bottle_neck})."
        gat_dim = dim // bottle_neck
        self.use_checkpoint = use_checkpoint
        self.norm1 = nn.LayerNorm(dim)
        self.j_conv = nn.Sequential(nn.Linear(dim, gat_dim), GAT(gat_dim, gat_dim, mode='joint'), nn.Linear(gat_dim, dim))
        self.j_alpha = nn.Sequential(nn.Linear(dim, gat_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(gat_dim, 2), nn.Softmax(dim=-1))
        self.drop1 = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        self.to_bone = nn.Linear(j_graph.num_nodes, b_graph.num_nodes)
        self.to_joint = nn.Linear(b_graph.num_nodes, j_graph.num_nodes)
        self.b_conv = nn.Sequential(nn.Linear(dim, gat_dim), GAT(gat_dim, gat_dim, mode='bone'), nn.Linear(gat_dim, dim))
        self.b_alpha = nn.Sequential(nn.Linear(dim, gat_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(gat_dim, 2), nn.Softmax(dim=-1))
        self.drop2 = nn.Dropout(drop)

    def forward(self, x:torch.Tensor):
        if self.training and self.use_checkpoint:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

    
    def _forward_impl(self, x:torch.Tensor):
        # Consider x.shape: [B, T, J, C].
        x = self.norm1(x)
        x0 = x
        x = self.joint_forward(x, x0)
        x = self.bone_forward(x, x0)
        return x
    
    def joint_forward(self, x:torch.Tensor, x0:torch.Tensor):
        x = self.j_conv(x)
        x = self.drop1(x)
        skip_rate = self.j_alpha(x + x0)
        x = skip_rate[..., 0:1] * x0 + skip_rate[..., 1:2] * x
        return x
    
    def bone_forward(self, x:torch.Tensor, x0:torch.Tensor):
        x = x.transpose(2, 3) # [B,T,C,J]
        x = self.to_bone(x)
        x = x.transpose(2, 3)
        x = self.norm2(x)
        x = self.b_conv(x)
        x = x.transpose(2, 3)
        x = self.to_joint(x)
        x = x.transpose(2, 3)
        x = self.drop2(x)
        skip_rate = self.b_alpha(x + x0)
        x = skip_rate[..., 0:1] * x0 + skip_rate[..., 1:2] * x
        return x




class GAT(nn.Module):
    def __init__(self, dim_in:int, dim_out:int, n_heads: int = 8, qkv_bias=False, a_scale:int=2, mode:str='joint'):
        super().__init__()
        assert dim_in % n_heads == 0, "dim_in must be divisible by n_heads"
        self.dim_h = dim_in // n_heads
        self.h = n_heads
        self.w_qkv = nn.Linear(dim_in, (2*a_scale+1)*dim_in, bias=qkv_bias) # (2C + 2C + C)
        self.a = nn.Linear(a_scale*self.dim_h, 1, bias=False)
        self.proj = nn.Linear(dim_in, dim_out)
        self.a_scale = a_scale
        self.g = b_graph if mode == 'bone' else j_graph
    
    def forward(self, x:torch.Tensor):
        B, T, J, C = x.shape
        A = self.a_scale*self.dim_h
        # Let start_node[i] = start node of i-th edge.
        start_node, end_node = self.g.edge_index[0], self.g.edge_index[1]
        qkv:torch.Tensor = self.w_qkv(x)
        qkv = qkv.view(B,T,J,self.h,2*A+self.dim_h)
        q, k, v = torch.split(qkv, split_size_or_sections=[A,A,self.dim_h], dim=-1) # [B,T,J,H, ?*dH]
        z = q[..., start_node,:,:] + k[..., end_node,  :,:] # [B,T,E,H,2dH]
        z = F.leaky_relu_(z, negative_slope=0.2)
        z:torch.Tensor = self.a(z).squeeze(-1) # [B,T,E,H]
        z = torch.exp(z - z.amax(dim=2, keepdim=True))
        sigma = x.new_zeros(B,T,J,self.h)
        sigma = sigma.index_add(dim=2, index=start_node, source=z) # [B,T,J,H]
        attn = z / sigma[..., start_node, :] # [B,T,E,H]
        attn = attn.transpose(2, 3) # [B,T,H,E]
        dense_attn = x.new_zeros(B,T,self.h,J,J)
        dense_attn[..., start_node, end_node] = attn # [B,T,H,J,J]
        v = v.transpose(2, 3) # [B,T,H,J,dH]
        v = dense_attn @ v # [B,T,H,J,dH]
        v = v.transpose(2, 3).reshape(B,T,J,C)
        v = self.proj(v)
        return F.relu(v)



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
    #profile_gat()
    test_many_modules()
    #profile_skip_gat()
