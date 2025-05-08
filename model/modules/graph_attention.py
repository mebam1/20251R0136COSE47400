import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.mlp import MLP

class CachedGraph():
    def __init__(self, adj:torch.Tensor, num_nodes:int):
        self.num_nodes = num_nodes
        self.adj = adj

    @staticmethod
    @torch.no_grad()
    def build_human_graph(joint_or_bone:str):
        if joint_or_bone not in ['joint', 'bone']:
            raise ValueError("joint_or_bone must be 'joint' or 'bone'")
        device ='cuda' if torch.cuda.is_available() else 'cpu'

        if joint_or_bone == 'joint':
            h36m_joint_edge_tensor = torch.tensor([
                [
                    0,0,0,1,1,2,2,3,4,4,5,5,6,7,7,8,8,8,8,9,9,10,11,11,12,12,13,14,14,15,15,16
                ],
                [
                    1,4,7,0,2,1,3,2,0,5,4,6,5,0,8,7,9,11,14,8,10,9,8,12,11,13,12,8,15,14,16,15
                ]
                ], device=device, dtype=torch.int)
            adj = torch.zeros((17, 17), device=device, dtype=torch.int8)
            adj[h36m_joint_edge_tensor[0], h36m_joint_edge_tensor[1]] = 1
            adj[h36m_joint_edge_tensor[1], h36m_joint_edge_tensor[0]] = 1
            adj.fill_diagonal_(1)
            return CachedGraph(adj, 17)
        else:
            bone_edge_tensor = torch.tensor([
                [
                    0,0,0,1,1,2,3,3,3,4,4,5,6,6,6,7,7,7,7,8,8,8,8,9,10,10,10,10,11,11,12,13,13,13,13,14,14,15
                ],
                [
                    1,3,6,0,2,1,0,4,6,3,5,4,0,3,7,6,8,10,13,7,9,10,13,8,7,8,11,13,10,12,11,7,8,10,14,13,15,14
                ]
                ], device=device, dtype=torch.int)
            adj = torch.zeros((16, 16), device=device, dtype=torch.int8)
            adj[bone_edge_tensor[0], bone_edge_tensor[1]] = 1
            adj[bone_edge_tensor[1], bone_edge_tensor[0]] = 1
            adj.fill_diagonal_(1)
            return CachedGraph(adj, 16)

j_graph=CachedGraph.build_human_graph('joint')
b_graph=CachedGraph.build_human_graph('bone')

class SpatialAPPNP(nn.Module):
    def __init__(self, dim:int, j_or_e:CachedGraph, appnp_iter:int=1, drop:float=0.0):
        super().__init__()
        self.human_graph = j_or_e
        self.appnp1 = GATAppnp(dim, appnp_iter)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        # B * T are spatial batch dimension.
        # Consider x.shape: [B*T, J, C].
        x = self.appnp1(x, self.human_graph)
        x = self.drop(x)
        return x

class SpatialBlock(nn.Module):
    def __init__(self, dim, appnp_iter:int=1, drop:float=0.0):
        super().__init__()
        self.s1 = SpatialAPPNP(dim, j_graph, appnp_iter=appnp_iter, drop=drop)
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 2, dim)
        self.s2 = SpatialAPPNP(dim, j_graph, appnp_iter=appnp_iter, drop=drop)

    def forward(self, x: torch.Tensor):
        # Consider x.shape: [B,T,J,C].
        B,T,J,C = x.shape
        x = x.view(-1, J, C)
        x = self.s1(x)
        y = self.norm(x)
        y = self.mlp(y)
        x = x + y
        x = self.s2(x)
        x = x.view(B, T, J, C)
        return x

class GATAppnp(nn.Module):
    def __init__(self, dim:int, n_iter:int):
        super().__init__()
        self.alpha = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.conv = GAT(dim)
        self.n_iter = n_iter

    def forward(self, x:torch.Tensor, g:CachedGraph):
        # B * T are spatial batch dimension.
        # Consider x.shape: [B*T, J, C].
        x0 = x

        for _ in range(self.n_iter):
            x = self.norm(x)
            skip_rate = F.sigmoid(self.alpha(x))
            x = (1.0 - skip_rate) * self.conv(x, g) + skip_rate * x0
        return x

class GAT(nn.Module):
    def __init__(self, dim:int, n_heads: int = 2):
        super().__init__()
        # num_param = 5*C^2 + 2*C + etc
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim_h = dim // n_heads
        self.h = n_heads
        self.w1 = nn.Linear(dim, 2*dim)
        self.w2 = nn.Linear(dim, 2*dim)
        self.act1 = nn.GELU()
        self.proj_to_score = nn.Linear(2*self.dim_h,1)
        self.v = nn.Linear(dim, dim)
        self.act2 = nn.GELU()

    def forward(self, x:torch.Tensor, g:CachedGraph):
        # B * T are spatial batch dimension.
        # Consider x.shape: [B*T, J, C].
        BT, J, C = x.shape
        attn_mask = (g.adj == 0) # mask which considers human structure

        l = self.w1(x)
        r = self.w2(x)
        l = l.view(BT, J, self.h, 2 * self.dim_h)
        l = l.transpose(1, 2) # shape: [BT, H, J, 2dH]
        l = l.unsqueeze(3) # shape: [BT, H, J, 1, 2dH]
        r = r.view(BT, J, self.h, 2 * self.dim_h)
        r = r.transpose(1, 2)
        r = r.unsqueeze(2) # shape: [BT, H, 1, J, 2dH]        

        s = self.act1(l+r) # shape: [BT, H, J, J, 2dH]     

        s = self.proj_to_score(s) # [BT, H, J, J, 1]
        s = s.squeeze(-1) # [B*T, H, J, J]
        s = s.masked_fill(attn_mask, -1e9)
        s = F.softmax(s, dim=-1) # [B*T, H, J, J]

        x = self.v(x) # shape: [BT, J, C]
        x = x.view(BT, J, self.h, self.dim_h)
        x = x.transpose(1, 2) # shape: [BT, H, J, dH]
        x = s @ x # shape: [BT, H, J, dH]
        x = self.act2(x)
        x = x.transpose(1, 2) # shape: [BT, J, H, dH]
        x = x.reshape(BT, J, C)
        return x


#==================#
#  UNIT TEST CODE  #
#==================#

def show_mem(tag=""):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserv = torch.cuda.memory_reserved() / 1024**2
    print(f"[{tag}] allocated = {alloc:.1f} MB | reserved = {reserv:.1f} MB")

def test_spatial_graph_attention():
    # Create sample data
    edge_index = torch.tensor([[0,1], [1,2]], dtype=torch.long)
    
    # Initialize model
    B, T, J, C = 8, 81, 17, 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SpatialBlock(dim=32, appnp_iter=2, drop=0.3)
    model.to(device)
    
    # Test forward pass
    x = torch.randn(B, T, J, C, device=device)
    show_mem(f"before ---")
    output = model(x)
    show_mem(f"after ---")
    assert not torch.allclose(output, x), "Model not modifying input"

if __name__ == '__main__':
    test_spatial_graph_attention()
