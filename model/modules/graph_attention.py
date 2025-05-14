import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as S
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
                ], device=device, dtype=torch.int64)
            
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
                ], device=device, dtype=torch.int64)
            #x = x[:, (x[0] < x[1])]
            x = torch.cat([x, torch.arange(0, 16, step=1,device=device, dtype=torch.int64).expand(2, -1)], dim=1)
            return CachedGraph(x, 16)

j_graph=CachedGraph.build_human_graph('joint')
b_graph=CachedGraph.build_human_graph('bone')



class GATAPPNP(nn.Module):
    def __init__(self, dim:int, appnp_iter:int=4, mlp_ratio:float=4.0, drop:float=0.0):
        super().__init__()
        self.upscale_to_bone = nn.Linear(j_graph.num_nodes, b_graph.num_nodes)
        self.downscale_to_joint = nn.Linear(b_graph.num_nodes, j_graph.num_nodes)
        self.alpha = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2)
        )

        self.bone_gat = SkipableGAT(dim, appnp_iter, drop=drop)
        self.joint_gat = SkipableGAT(dim, appnp_iter, drop=drop)

    def forward(self, x: torch.Tensor):
        bone = self.bone_forward(x)
        joint = self.joint_gat(x, j_graph)
        fusion_rate = F.softmax(self.alpha(torch.cat([bone, joint], dim=-1)), dim=-1)
        x = fusion_rate[..., 0:1] * bone + fusion_rate[..., 1:2] * joint
        return x
    
    def bone_forward(self, x: torch.Tensor):
        residual = x
        x = self.upscale_to_bone(x.transpose(2, 3)).transpose(2, 3) # [B,T,J + 1,C]
        x = self.bone_gat(x, b_graph)
        x = self.downscale_to_joint(x.transpose(2, 3)).transpose(2, 3) # [B,T,J,C]
        return x + residual



class SkipableGAT(nn.Module):
    def __init__(self, dim:int, n_iter:int, drop:float=0.0):
        super().__init__()
        '''
        self.alpha = nn.ModuleList([nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2)
        )
        for _ in range(n_iter)])
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_iter)])
        self.conv = nn.ModuleList([GAT(dim, dim) for _ in range(n_iter)])
        '''
        
        self.n_iter = n_iter
        self.norm = nn.LayerNorm(dim)
        self.norm_init = nn.LayerNorm(dim)
        scale = 1
        self.conv = GAT(dim // scale, dim // scale)
        self.alpha = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 2))
        self.down_scale = nn.Linear(dim, dim // scale, bias=False)
        self.up_scale = nn.Linear(dim // scale, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)


    def forward(self, x:torch.Tensor, g:CachedGraph):
        # B * T are spatial batch dimension.
        # Consider x.shape: [B, T, J, C].
        x0 = self.norm_init(x)

        for i in range(self.n_iter):
            x = self.norm(x)
            x = self.conv(x, g)
            #skip_rate = F.softmax(self.alpha(x + x0), dim=-1)
            #x = skip_rate[..., 0:1] * x0 + skip_rate[..., 1:2] * self.drop(self.up_scale(self.conv(self.down_scale(x), g)))

        return self.proj(x)


class GAT(nn.Module):
    def __init__(self, dim_in:int, dim_out:int, n_heads: int = 8, qkv_bias=False):
        super().__init__()
        assert dim_in % n_heads == 0, "dim_in must be divisible by n_heads"
        self.dim_h = dim_in // n_heads
        self.h = n_heads
        #self.qk_scale = self.dim_h ** -0.5 # 1 / sqrt(dH)
        self.w_qkv = nn.Linear(dim_in, 5*dim_in, bias=qkv_bias) # (2C + 2C + C)
        self.a = nn.Linear(2*self.dim_h, 1, bias=False)
        self.proj = nn.Linear(dim_in, dim_out)
        self.relu_ = nn.ReLU(inplace=True)
    
    def forward(self, x:torch.Tensor, g:CachedGraph):
        B, T, J, C = x.shape
        # Let start_node[i] = start node of i-th edge.

        start_node, end_node = g.edge_index[0], g.edge_index[1]
        qkv:torch.Tensor = self.w_qkv(x)
        qkv = qkv.view(B,T,J,self.h,5*self.dim_h).transpose(2, 3) # [B,T,J,H,5dH]

        
        q, k, v = qkv[..., :2*self.dim_h], qkv[..., 2*self.dim_h:4*self.dim_h], qkv[..., 4*self.dim_h:]
        
        with profiler.record_function("!Optim1"):
            qe = q.index_select(dim=2, index=start_node) # [B,T,E,H,2dH]
            print(qe.shape)
            quit()
        ke = k.index_select(dim=2, index=end_node)
        ve = v.index_select(dim=2, index=end_node)

        z = F.leaky_relu_(qe + ke, negative_slope=0.2)
        z = self.a(z) # [B,T,H,E]
        z = z.squeeze(-1)
        z = torch.exp(z - torch.max(z, dim=-1, keepdim=True).values)
        sigma = x.new_zeros(B,T,self.h,J)
        sigma.index_add_(dim=-1, index=start_node, source=z)
        attn = z / sigma[..., start_node] # [B,T,H,E]

        with profiler.record_function("!Optim2."):
            attn.unsqueeze_(-1)
            weighted_v = attn * ve # [B,T,H,E,dH]

        out = x.new_zeros(B,T,self.h,J,self.dim_h)
        out.index_add_(dim=-2, index=start_node, source=weighted_v)
        out.transpose_(2, 3)
        out = out.view(B,T,J,C)
        out = self.proj(out)
        return F.relu_(out)



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
    model = GAT(C,C,1)
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
    model = nn.ModuleList([GAT(C, C, 1, False) for _ in range(16)])
    model.to(device)

    x = torch.randn(B, T, J, C, device=device)
    show_mem('---before---')
    for m in model:
        x = m(x, j_graph)
    show_mem('---after---')

if __name__ == '__main__':
    profile_gat()
